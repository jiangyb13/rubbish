pt_path   = 'attn_maps/attn_sample00_tensors.pt'
  save_dir  = 'attn_maps'                                                                                                              
  T_vid, H_vid, W_vid = 121, 720, 1280
                                                                                                                                       
  # ---- 加载 ----
  maps = torch.load(pt_path, map_location='cpu')  # list of 50, each [2, 20, 57600]                                                    
                                                                                                                                       
  os.makedirs(save_dir, exist_ok=True)
                                                                                                                                       
  for t_idx, attn in enumerate(maps):                                                                                                  
      # 取 cond，对 20 头求平均
      attn = attn[0].mean(dim=0)          # [57600]                                                                                    
      attn = attn.reshape(16, 45, 80)     # [T_lat, H_patch, W_patch]                                                                  
   
      # 上采样到视频分辨率                                                                                                             
      attn = F.interpolate(
          attn.unsqueeze(0).unsqueeze(0).float(),  # [1,1,16,45,80]                                                                    
          size=(T_vid, H_vid, W_vid),
          mode='trilinear',                                                                                                            
          align_corners=False,                                                                                                         
      ).squeeze()  # [121, 720, 1280]
                                                                                                                                       
      # 归一化到 [0, 255]
      vmin, vmax = attn.min(), attn.max()
      attn_np = ((attn - vmin) / (vmax - vmin + 1e-8) * 255).numpy().astype(np.uint8)                                                  
   
      # 保存热力图视频                                                                                                                 
      save_path = os.path.join(save_dir, f't{t_idx:03d}.mp4')
      writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), 24, (W_vid, H_vid))                                         
      for i in range(T_vid):                                                                                                           
          writer.write(cv2.applyColorMap(attn_np[i], cv2.COLORMAP_JET))
      writer.release()                                                                                                                 
      print(f'Saved {save_path}')
                                                                                                                                       
  print('Done.')  
