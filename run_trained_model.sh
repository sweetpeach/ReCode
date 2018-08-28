output="../../files"
models="../../files"
device="cpu"

if [ "$1" == "hs" ]; then
	# hs dataset
	echo "run trained model for hs"
	dataset="../../files/aligned_hs.bin"
	#model="model.hs.npz"
	model="model.hs.ours.best_acc.npz"
	commandline="-decode_max_time_step 750 -rule_embed_dim 128 -node_embed_dim 64 -enable_retrieval -retrieval_factor 3 -max_retrieved_sentences 3"
	datatype="hs"
else
	# django dataset
	echo "run trained model for django"
	dataset="../../files/django.cleaned.dataset.freq5.par_info.refact.space_only.bin"
	model="model.django.npz"
	commandline="-rule_embed_dim 128 -node_embed_dim 64 -enable_retrieval -retrieval_factor 3 -max_retrieved_sentences 10"
	datatype="django"
fi

# decode the test set and save the nbest decoding results
THEANO_FLAGS="mode=FAST_RUN,device=${device},floatX=float32" python2 code_gen.py \
-data_type ${datatype} \
-data ${dataset} \
-output_dir ${output} \
-model ${models}/${model} \
${commandline} \
decode \
-saveto ${output}/${model}.decode_results.test.bin

# evaluate the decoding result
python2 code_gen.py \
-data_type ${datatype} \
-data ${dataset} \
-output_dir ${output} \
evaluate \
-input ${output}/${model}.decode_results.test.bin | tee ${output}/${model}.decode_results.test.log
