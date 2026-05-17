pt_path        = 'attn_maps/attn_sample00_tensors.pt'
  save_dir       = 'attn_maps'                                                                                                         
  H_ref_patch, W_ref_patch = 80, 45   # 竖屏；横屏改 (45, 80)                                                                          
  H_ref, W_ref             = 1280, 720 # 竖屏；横屏改 (720, 1280)                                                                      
                                                                                                                                       
  # ---- 加载 ----                                                                                                                     
  maps = torch.load(pt_path, map_location='cpu')                                                                                       
  # maps: list of 50, 每个 [B=2, heads=20, T_lat=16, T_ref=3600]
                                                                                                                                       
  os.makedirs(save_dir, exist_ok=True)                                                                                                 
                                                                                                                                       
  for t_idx, attn_map in enumerate(maps):                                                                                              
      # 1. 对 heads 求平均，取 cond（index 0）
      avg = attn_map.mean(dim=1)   # [2, 16, 3600]
      avg = avg[0].float()         # [16, 3600]                                                                                        
                                                                                                                                       
      # 2. reshape T_ref → ref 图片 patch grid                                                                                         
      avg = avg.reshape(16, H_ref_patch, W_ref_patch)  # [16, 80, 45]                                                                  
                                                                                                                                       
      # 3. upsample 到 ref 图片分辨率
      avg = F.interpolate(                                                                                                             
          avg.unsqueeze(0).unsqueeze(0),   # [1, 1, 16, 80, 45]
          size=(16, H_ref, W_ref),                                                                                                     
          mode='trilinear',
          align_corners=False,                                                                                                         
      ).squeeze().numpy()  # [16, 1280, 720]                                                                                           
   
      # 4. 归一化到 [0, 255]                                                                                                           
      vmin, vmax = avg.min(), avg.max()
      avg = ((avg - vmin) / (vmax - vmin + 1e-8) * 255).astype(np.uint8)                                                               
                                                                                                                                       
      # 5. 保存视频（16帧，每帧是 ref 图片热力图）
      save_path = os.path.join(save_dir, f't{t_idx:03d}.mp4')                                                                          
      writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), 8, (W_ref, H_ref))                                          
      for i in range(16):                                                                                                              
          writer.write(cv2.applyColorMap(avg[i], cv2.COLORMAP_JET))                                                                    
      writer.release()                                                                                                                 
      print(f'Saved {save_path}')  
