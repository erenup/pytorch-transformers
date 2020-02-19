cuda_num=$1
pipeline_path=data/hotpot/small
eval_file=dev.json
eval_file=${pipeline_path}/${eval_file}
output_dir_name=dev_ranked
para_num=4
qa_model=../transformers/models_roberta/develop_qa_model
yes_no_model=../transformers/models_roberta/develop_yes_no_model
sf_model=../transformers/models_roberta/develop_sf_model
pre_train_model_type=roberta
python processing/preprocessing.py \
	--hotpot_file  ${eval_file} \
	--converted_keys ranked \
	--max_para_num 4 \
	--output_dir ${output_dir_name} \
	--split 1 \
	--squad_key selected \
	--mode test \
	--sub_key ranked \
	--para_score_th 1000
CUDA_VISIBLE_DEVICES=${cuda_num} python training_code/hotpot_qa.py \
	--model_type ${pre_train_model_type} \
	--model_name_or_path ${qa_model}  \
	--do_eval \
	--do_lower_case \
	--train_file data/hotpotqq/pipeline/train.json.squad.selected \
	--predict_file ${pipeline_path}/${output_dir_name}_${para_num}/squad/dev.json.squad.selected  \
	--learning_rate 2e-5 \
	--warmup_steps 500 \
	--num_train_epochs 2 \
	--max_seq_length 512 \
	--doc_stride 128 \
	--output_dir ${qa_model} \
	--per_gpu_eval_batch_size=8 \
	--per_gpu_train_batch_size=6 \
	--version_2_with_negative \
	--overwrite_cache \
	--threads 48 \
	--save_steps 1000 \
	--overwrite_output_dir \
	--gradient_accumulation_steps 1
CUDA_VISIBLE_DEVICES=${cuda_num} python training_code/hotpot_cls.py \
	--model_type ${pre_train_model_type} \
	--model_name_or_path ${yes_no_model} \
	--task_name hotpot_yes_no \
	--do_eval \
	--do_lower_case \
	--data_dir ${pipeline_path}/${output_dir_name}_${para_num}/yes_no \
	--max_seq_length 250 \
	--per_gpu_eval_batch_size=32 \
	--per_gpu_train_batch_size=3 \
	--gradient_accumulation_steps 1 \
	--learning_rate 3e-5 \
	--num_train_epochs 2 \
	--output_dir ${yes_no_model} \
	--save_steps 10000 \
	--overwrite_cache
python evaluate/hotpot_pipline.py \
	--ranked_file  ${eval_file} \
	--ranked_qa_result ${qa_model}/predictions_.json \
	--yes_no_cls_input ${pipeline_path}/${output_dir_name}_${para_num}/yes_no/dev.json.para.yes_no \
	--yes_no_result_file ${pipeline_path}/${output_dir_name}_${para_num}/yes_no/eval_results.json \
	--option 1 \
	--sf_input ${pipeline_path}/yes_no/pred_sf_para/dev.json.para.sf \
	--sf_result_file ${pipeline_path}/yes_no/pred_sf_para/eval_results.json \
	--qa_result ${pipeline_path}/${output_dir_name}_${para_num}/yes_no/.qa.result \
	--gold_file  ${eval_file} \
	--sf_th 0.0 
CUDA_VISIBLE_DEVICES=${cuda_num} python training_code/hotpot_cls.py \
	--model_type ${pre_train_model_type} \
	--model_name_or_path ${sf_model} \
	--task_name hotpot_sf \
	--do_eval \
	--do_lower_case \
	--data_dir ${pipeline_path}/${output_dir_name}_${para_num}/yes_no/pred_sf_para \
	--max_seq_length 250 \
	--per_gpu_eval_batch_size=32 \
	--per_gpu_train_batch_size=3 \
	--gradient_accumulation_steps 1 \
	--learning_rate 3e-5 \
	--num_train_epochs 2 \
	--output_dir ${sf_model} \
	--save_steps 10000 \
	--overwrite_cache
python evaluate/hotpot_pipline.py \
	--ranked_file  ${eval_file} \
	--ranked_qa_result ${qa_model}/predictions_.json \
	--yes_no_cls_input ${pipeline_path}/${output_dir_name}_${para_num}/yes_no/dev.json.para.yes_no \
	--yes_no_result_file ${pipeline_path}/${output_dir_name}_${para_num}/yes_no/eval_results.json \
	--sf_input ${pipeline_path}/${output_dir_name}_${para_num}/yes_no/pred_sf_para/dev.json.para.sf \
	--sf_result_file ${pipeline_path}/${output_dir_name}_${para_num}/yes_no/pred_sf_para/eval_results.json \
	--qa_result ${pipeline_path}/${output_dir_name}_${para_num}/yes_no/.qa.result \
	--gold_file  ${eval_file} \
	--sf_th 0.0 \
	--option 2
