# Add parent directory to python path to access lightning_base.py and utils.py
export PYTHONPATH="../":"${PYTHONPATH}"
python finetune.py \
--dataset=cnn_tiny \
--model_name=bart_tiny \
--experiment=bart_tiny_default \
--learning_rate=3e-5 \
--train_batch_size=2 \
--eval_batch_size=2 \
--num_train_epochs=1  \
--n_gpu=0 \
--do_train $@
