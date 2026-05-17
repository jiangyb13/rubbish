import cv2                                                                                                                           
  import glob                                                                                                                          
  import os                                                                                                                            
                                                                                                                                       
  input_dir  = 'attn_maps'                                                                                                             
  output_dir = 'attn_maps_gray'
  os.makedirs(output_dir, exist_ok=True)                                                                                               
   
  for input_path in sorted(glob.glob(os.path.join(input_dir, '*.mp4'))):                                                               
      output_path = os.path.join(output_dir, os.path.basename(input_path))
                                                                                                                                       
      cap = cv2.VideoCapture(input_path)
      fps    = cap.get(cv2.CAP_PROP_FPS)
      width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))                                                                                  
      height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                                                                                                                       
      writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))                                     
      while True:
          ret, frame = cap.read()                                                                                                      
          if not ret:
              break
          gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
          writer.write(cv2.merge([gray, gray, gray]))                                                                                  
   
      cap.release()                                                                                                                    
      writer.release()
      print(f'Saved {output_path}')
