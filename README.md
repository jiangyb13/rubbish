Traceback (most recent call last):
  File "/home/ma-user/anaconda3/envs/PyTorch-2.6.0/lib/python3.11/site-packages/flask/app.py", line 2213, in __call__
    return self.wsgi_app(environ, start_response)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ma-user/anaconda3/envs/PyTorch-2.6.0/lib/python3.11/site-packages/flask/app.py", line 2193, in wsgi_app
    response = self.handle_exception(e)
               ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ma-user/anaconda3/envs/PyTorch-2.6.0/lib/python3.11/site-packages/flask/app.py", line 2190, in wsgi_app
    response = self.full_dispatch_request()
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ma-user/anaconda3/envs/PyTorch-2.6.0/lib/python3.11/site-packages/flask/app.py", line 1486, in full_dispatch_request
    rv = self.handle_user_exception(e)
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ma-user/anaconda3/envs/PyTorch-2.6.0/lib/python3.11/site-packages/flask/app.py", line 1484, in full_dispatch_request
    rv = self.dispatch_request()
         ^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ma-user/anaconda3/envs/PyTorch-2.6.0/lib/python3.11/site-packages/flask/app.py", line 1469, in dispatch_request
    return self.ensure_sync(self.view_functions[rule.endpoint])(**view_args)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/huanan/code/jwx1416454/HUAWEI_CrossPairDataset/flask/app.py", line 201, in api_shot_thumbnail
    return send_file(probe_path, mimetype="image/jpeg")
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ma-user/anaconda3/envs/PyTorch-2.6.0/lib/python3.11/site-packages/flask/helpers.py", line 504, in send_file
    return werkzeug.utils.send_file(  # type: ignore[return-value]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ma-user/anaconda3/envs/PyTorch-2.6.0/lib/python3.11/site-packages/werkzeug/utils.py", line 428, in send_file
    stat = os.stat(path)
           ^^^^^^^^^^^^^
