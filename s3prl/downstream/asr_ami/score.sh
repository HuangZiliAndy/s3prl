export PATH="/scratch4/lgarci27/hzili1/anaconda3/envs/espnet/bin:$PATH"

espnet_dir="/scratch4/lgarci27/hzili1/workspace/espnet"
export PYTHONPATH="${espnet_dir}:$PYTHONPATH"
export PATH=$espnet_dir/tools/sctk/bin:${PATH}
_opts="--token_type word --non_linguistic_symbols none --remove_non_linguistic_symbols true"
score_opts=
dset="test"

exp_dir=$1
use_lm=$2
test_dir=$3

if $use_lm;
then
  ref_file="${dset}-LM-ref.ark"
  hyp_file="${dset}-LM-hyp.ark"
  ref_trn_file="${dset}-LM-ref.trn"
  hyp_trn_file="${dset}-LM-hyp.trn"
else
  ref_file="${dset}-noLM-ref.ark"
  hyp_file="${dset}-noLM-hyp.ark"
  ref_trn_file="${dset}-noLM-ref.trn"
  hyp_trn_file="${dset}-noLM-hyp.trn"
fi

paste \
	<(<"${exp_dir}/${ref_file}" \
                        python3 ${espnet_dir}/espnet2/bin/tokenize_text.py  \
                            -f 2- --input - --output - \
                            --cleaner "none" \
                            ${_opts} \
                            ) \
                    <(<"${test_dir}/utt2spk" awk '{ print "(" $2 "-" $1 ")" }') \
                        >"${exp_dir}/${ref_trn_file}"

paste \
	<(<"${exp_dir}/${hyp_file}" \
                        python3 ${espnet_dir}/espnet2/bin/tokenize_text.py  \
                            -f 2- --input - --output - \
                            --cleaner "none" \
                            ${_opts} \
                            ) \
                    <(<"${test_dir}/utt2spk" awk '{ print "(" $2 "-" $1 ")" }') \
                        >"${exp_dir}/${hyp_trn_file}"

sclite \
                ${score_opts} \
                -r "${exp_dir}/${ref_trn_file}" trn \
                -h "${exp_dir}/${hyp_trn_file}" trn \
                -i rm -o all stdout > "${exp_dir}/result_${dset}.txt"

grep -e Avg -e SPKR -m 2 "${exp_dir}/result_${dset}.txt"
