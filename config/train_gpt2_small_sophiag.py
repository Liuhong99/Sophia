wandb_log = True
wandb_project = 'sophia'
wandb_run_name='gpt2-small-sophiag-100k'

# these make the total batch size be ~0.5M
# 8 batch size * 1024 block size * 6 gradaccum * 10 GPUs = 491,520
batch_size = 8
block_size = 1024
gradient_accumulation_steps = 6
total_bs = 480

n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False

# this makes total number of tokens be 300B
max_iters = 100000
lr_decay_iters = 100000

# eval stuff
eval_interval = 1000
eval_iters = 200
log_interval = 10

# optimizer
optimizer_name = 'sophiag'
learning_rate = 7e-4 # max learning rate
weight_decay = 2e-1
beta1 = 0.965
beta2 = 0.99
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
min_lr = 1.5e-5 
rho = 0.05
interval = 10

compile = True

out_dir = 'out_small_sophiag_100k'
