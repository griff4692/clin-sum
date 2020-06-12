# Add parent directory to python path to access lightning_base.py
export PYTHONPATH="../":"${PYTHONPATH}"

python finetune.py \
--dataset=cnn_dm \
--model_name=bart \
--experiment=bart_default \
--learning_rate=3e-5 \
--train_batch_size=4 \
--eval_batch_size=4 \
--do_train  $@
