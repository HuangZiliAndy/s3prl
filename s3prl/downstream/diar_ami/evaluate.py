import os
import torch
import argparse
import numpy as np
from s3prl.downstream.runner import Runner
from pyannote.audio.core.model import Model
from pyannote.audio.core.task import Task
from pyannote.audio.pipelines.speaker_diarization import SpeakerDiarization
from pyannote.audio.core.task import Specifications, Problem, Resolution
from pyannote.audio import Pipeline
from typing import Optional
from torch.nn.utils.rnn import pad_sequence
from s3prl.downstream.diar_ami.dataset import DiarizationDataset

parser = argparse.ArgumentParser(description='Evaluate diarization model')
parser.add_argument('ckpt_path', type=str, help='checkpoint path')
parser.add_argument('test_dir', type=str, help='test directory')
parser.add_argument('output_dir', type=str, help='output directory')
parser.add_argument('--channel', type=str, default='0', help='audio channel')
parser.add_argument('--normalize', type=int, default=0, help='whether to normalize audio')
parser.add_argument('--min_cluster_size', type=int, default=12, help='minimum cluster size')
parser.add_argument('--cluster_thres', type=float, default=0.7045654963945799, help='clustering threshold')
parser.add_argument('--segmentation_thres', type=float, default=0.5, help='segmentation threshold')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class S3PRLModel(pl.LightningModule):
    def __init__(
        self,
        upstream,
        featurizer,
        downstream,
        sample_rate: int = 16000,
        num_channels: int = 1,
        task: Optional[Task] = None,
    ):
        super().__init__(sample_rate=sample_rate, num_channels=num_channels, task=task)

        self.upstream = upstream
        self.featurizer = featurizer
        self.downstream = downstream

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Pass forward

        Parameters
        ----------
        waveforms : (batch, channel, sample)

        Returns
        -------
        scores : (batch, frame, classes)
        """
        if waveforms.size(1) == 1:
            wavs = [(wav[0, :]).to(device) for wav in waveforms]
        else:
            wavs = [(wav[:, :]).transpose(0, 1).to(device) for wav in waveforms]
        print("wavs", [wav.size() for wav in wavs])
        with torch.no_grad():
            features = self.upstream.model(wavs)
            features = self.featurizer.model(wavs, features)
            #print("features", [feat.size() for feat in features])
            features = pad_sequence(features, batch_first=True)
            #print("features", features.size())
            prediction = torch.sigmoid(self.downstream.model.model(features))
            #print("prediction", prediction.size())
            #print("prediction", torch.max(prediction), torch.min(prediction))
        return prediction

def main():
    print("Loading {}".format(args.ckpt_path))
    ckpt = torch.load(args.ckpt_path, map_location='cpu')
    ckpt_args, ckpt_config = ckpt['Args'], ckpt['Config']
    ckpt_args.init_ckpt = args.ckpt_path
    runner = Runner(ckpt_args, ckpt_config)
    upstream = runner.upstream
    featurizer = runner.featurizer
    downstream = runner.downstream
    segmentation_model = S3PRLModel(upstream, featurizer, downstream, num_channels=len(args.channel.split(',')))
    segmentation_model.specifications = Specifications(
            problem=Problem.MULTI_LABEL_CLASSIFICATION,
            resolution=Resolution.FRAME,
            duration=10.0,
            min_duration=0.0,
            warm_up=(0.0, 0.0),
            classes=[f"speaker#{i+1}" for i in range(4)],
            powerset_max_classes=None,
            permutation_invariant=True,
        )

    pipeline = SpeakerDiarization(
        segmentation=segmentation_model,
        segmentation_step=0.1,
        embedding='pyannote/wespeaker-voxceleb-resnet34-LM',
        embedding_exclude_overlap=True,
        clustering='AgglomerativeClustering',
        embedding_batch_size=32,
        segmentation_batch_size=32,
        use_auth_token='YOUR_HF_TOKEN'
    )
    params = {'clustering': {'method': 'centroid', 'min_cluster_size': args.min_cluster_size, 'threshold': args.cluster_thres}, 'segmentation': {'min_duration_off': 0.0, 'threshold': args.segmentation_thres}}
    pipeline.instantiate(params)
    pipeline.to(device)

    ##pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token='YOUR_HF_TOKEN')
    ##pipeline.to(device)

    dataset = DiarizationDataset(args.test_dir, num_spks=5, frame_shift=320, channel=args.channel, normalize=args.normalize)
    for i, v in enumerate(dataset):
        audio, _, _, uttname = v
        #print('-' * 80)
        #print("{}/{}".format(i+1, len(dataset)))
        #print("audio", audio.shape)
        #print("uttname", uttname)

        audio = torch.from_numpy(audio).float()
        if len(audio.size()) == 1:
            audio = audio.unsqueeze(0)
        elif len(audio.size()) == 2:
            audio = audio.transpose(0, 1)

        diar_result = pipeline({"waveform": audio, "sample_rate": 16000, "uri":uttname})
        with open('{}/{}.rttm'.format(args.output_dir, uttname), 'w') as rttm_f:
            diar_result.write_rttm(rttm_f)
    return 0

if __name__ == '__main__':
    main()
