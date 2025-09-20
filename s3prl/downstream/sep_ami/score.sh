export PATH="/home/hzili1/anaconda3/envs/espnet_24_12_28/bin:$PATH"

espnet_dir="/export/c02/hzili1/workspace/espnet"
export PYTHONPATH="${espnet_dir}:$PYTHONPATH"
export PATH=$espnet_dir/tools/sctk/bin:${PATH}

exp_dir=$1
token_type=$2

_opts="--token_type ${token_type} --non_linguistic_symbols none --remove_non_linguistic_symbols true"
score_opts=

ref_file="ref.ark"
hyp_file="hyp.ark"
ref_trn_file="ref.trn"
hyp_trn_file="hyp.trn"

paste \
	<(<"${exp_dir}/${ref_file}" \
                        python3 ${espnet_dir}/espnet2/bin/tokenize_text.py  \
                            -f 2- --input - --output - \
                            --cleaner "none" \
                            ${_opts} \
                            ) \
                    <(<"${exp_dir}/utt2spk" awk '{ print "(" $2 "-" $1 ")" }') \
                        >"${exp_dir}/${ref_trn_file}"

paste \
	<(<"${exp_dir}/${hyp_file}" \
                        python3 ${espnet_dir}/espnet2/bin/tokenize_text.py  \
                            -f 2- --input - --output - \
                            --cleaner "none" \
                            ${_opts} \
                            ) \
                    <(<"${exp_dir}/utt2spk" awk '{ print "(" $2 "-" $1 ")" }') \
                        >"${exp_dir}/${hyp_trn_file}"

sclite \
                ${score_opts} \
                -r "${exp_dir}/${ref_trn_file}" trn \
                -h "${exp_dir}/${hyp_trn_file}" trn \
                -i rm -o all stdout > "${exp_dir}/result.txt"

grep -e Avg -e SPKR -e Sum -m 4 "${exp_dir}/result.txt"
