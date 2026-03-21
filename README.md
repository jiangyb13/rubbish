Traceback (most recent call last):
  File "/data/huanan/code/jwx1416454/HUAWEI_CrossPairDataset/main.py", line 107, in <module>
    mainForOneShot()
  File "/data/huanan/code/jwx1416454/HUAWEI_CrossPairDataset/main.py", line 76, in mainForOneShot
    pipeline.process_all()
  File "/data/huanan/code/jwx1416454/HUAWEI_CrossPairDataset/utils/one_shot.py", line 765, in process_all
    self.visualizer.render_tracking_video(
  File "/data/huanan/code/jwx1416454/HUAWEI_CrossPairDataset/utils/one_shot.py", line 562, in render_tracking_video
    writer.append_data(frame_rgb)
  File "/home/ma-user/anaconda3/envs/PyTorch-2.6.0/lib/python3.11/site-packages/imageio/v2.py", line 226, in append_data
    return self.instance.write(im, **self.write_args)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ma-user/anaconda3/envs/PyTorch-2.6.0/lib/python3.11/site-packages/imageio/plugins/tifffile_v3.py", line 224, in write
    self._fh.write(image, **kwargs)
TypeError: TiffWriter.write() got an unexpected keyword argument 'fps'
[ERROR] 2026-03-21-17:26:51 (PID:223902, Device:0, RankID:-1) ERR99999 UNKNOWN applicaiton exception
