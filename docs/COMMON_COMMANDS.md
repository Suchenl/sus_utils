# Common Commands

## Download from HuggingFace
### If too slow
- use mirror: export HF_ENDPOINT=https://hf-mirror.com

### template
huggingface-cli download <model_name> --local-dir <local_dir> --local-dir-use-symlinks False --resume-download

### use
huggingface-cli download ali-vilab/VACE-Annotators --local-dir /m2v_intern/chenyuzhuo03/MODELS/VideoGen/VACE/models --local-dir-use-symlinks False --resume-download

