# *TempDiff: Enhancing Temporal-awareness in Latent Diffusion for Real-world Video Super-resolution*

### Testing
Download the pretrained diffusion denoising U-net and video variational autoencoder from [[BaiduNetDisk]()]. Download the VideoLQ dataset following the links [here](https://github.com/ckkelvinchan/RealBasicVSR). Please update the ckpt_path, load_path and dataroot_gt paths in config files. 

Test on arbitrary size with chopping for VAE.
```
python scripts/vsr_val_ddpm_text_T_vqganfin_oldcanvas_tile.py \
  --config configs/unet/tempdiff_unet.yaml \
  --ckpt CKPT_PATH \
  --vqgan_ckpt VQGANCKPT_PATH \
  --seqs-path INPUT_PATH \
  --outdir OUT_DIR \
  --ddpm_steps 50 \
  --dec_w 1.0 \
  --colorfix_type adain \
  --select_idx 0 \
  --n_gpus 1
  
  

### Acknowledgement
This implementation largely depends on [StableSR](https://github.com/IceClear/StableSR). We thank the authors for the contribution.
