output="../../files/runs"
device="cpu"
models="../../files/"

if [ "$1" == "hs" ]; then
	# hs dataset
	echo "training hs dataset"
	dataset="hs.freq3.pre_suf.unary_closure.bin"
	model="model.hs.npz"
	commandline="-batch_size 10 -max_epoch 200 -valid_per_batch 280 -save_per_batch 280 -decode_max_time_step 750 -optimizer adadelta -rule_embed_dim 128 -node_embed_dim 64 -valid_metric bleu -enable_retrieval"
	datatype="hs"
else
	# django dataset
	echo "training django dataset"
	dataset="django.cleaned.dataset.freq5.par_info.refact.space_only.bin"
	model="model.hs.npz"
	commandline="-batch_size 10 -max_epoch 50 -valid_per_batch 4000 -save_per_batch 4000 -decode_max_time_step 100 -optimizer adam -rule_embed_dim 128 -node_embed_dim 64 -valid_metric bleu -enable_retrieval"
	datatype="django"
fi

THEANO_FLAGS="mode=FAST_RUN,device=${device},floatX=float32" python -u code_gen.py \
	-data_type ${datatype} \
	-data ../../files/${dataset} \
	-model ${models}/${model} \
	-output_dir ${output} \
	${commandline} \
	align
