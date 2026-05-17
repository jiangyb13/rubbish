aved attn map video to ./s3_id2_wo_ref2/I2V_ID_model.pth/id2/attn_maps/attn_sample00_t013.mp4
Saved attn map video to ./s3_id2_wo_ref2/I2V_ID_model.pth/id2/attn_maps/attn_sample00_t014.mp4
Saved attn map video to ./s3_id2_wo_ref2/I2V_ID_model.pth/id2/attn_maps/attn_sample00_t015.mp4
Saved attn map video to ./s3_id2_wo_ref2/I2V_ID_model.pth/id2/attn_maps/attn_sample00_t016.mp4
Saved attn map video to ./s3_id2_wo_ref2/I2V_ID_model.pth/id2/attn_maps/attn_sample00_t017.mp4
Saved attn map video to ./s3_id2_wo_ref2/I2V_ID_model.pth/id2/attn_maps/attn_sample00_t018.mp4
Saved attn map video to ./s3_id2_wo_ref2/I2V_ID_model.pth/id2/attn_maps/attn_sample00_t019.mp4
Saved attn map video to ./s3_id2_wo_ref2/I2V_ID_model.pth/id2/attn_maps/attn_sample00_t020.mp4
Saved attn map video to ./s3_id2_wo_ref2/I2V_ID_model.pth/id2/attn_maps/attn_sample00_t021.mp4
Saved attn map video to ./s3_id2_wo_ref2/I2V_ID_model.pth/id2/attn_maps/attn_sample00_t022.mp4
Saved attn map video to ./s3_id2_wo_ref2/I2V_ID_model.pth/id2/attn_maps/attn_sample00_t023.mp4
Traceback (most recent call last):
  File "/data/huanan/code/jwx1416454/ID_code_bak/scripts/inference_mmdit_i2v_ip_last_no_cfg_jsonl_prompt.py", line 683, in <module>
Traceback (most recent call last):
  File "/data/huanan/code/jwx1416454/ID_code_bak/scripts/inference_mmdit_i2v_ip_last_no_cfg_jsonl_prompt.py", line 683, in <module>
Traceback (most recent call last):
  File "/data/huanan/code/jwx1416454/ID_code_bak/scripts/inference_mmdit_i2v_ip_last_no_cfg_jsonl_prompt.py", line 683, in <module>
Traceback (most recent call last):
  File "/data/huanan/code/jwx1416454/ID_code_bak/scripts/inference_mmdit_i2v_ip_last_no_cfg_jsonl_prompt.py", line 683, in <module>
Traceback (most recent call last):
  File "/data/huanan/code/jwx1416454/ID_code_bak/scripts/inference_mmdit_i2v_ip_last_no_cfg_jsonl_prompt.py", line 683, in <module>
Traceback (most recent call last):
  File "/data/huanan/code/jwx1416454/ID_code_bak/scripts/inference_mmdit_i2v_ip_last_no_cfg_jsonl_prompt.py", line 683, in <module>
Traceback (most recent call last):
  File "/data/huanan/code/jwx1416454/ID_code_bak/scripts/inference_mmdit_i2v_ip_last_no_cfg_jsonl_prompt.py", line 683, in <module>
    main()    
main()  File "/data/huanan/code/jwx1416454/ID_code_bak/scripts/inference_mmdit_i2v_ip_last_no_cfg_jsonl_prompt.py", line 632, in main

  File "/data/huanan/code/jwx1416454/ID_code_bak/scripts/inference_mmdit_i2v_ip_last_no_cfg_jsonl_prompt.py", line 632, in main
    main()
  File "/data/huanan/code/jwx1416454/ID_code_bak/scripts/inference_mmdit_i2v_ip_last_no_cfg_jsonl_prompt.py", line 632, in main
    main()
  File "/data/huanan/code/jwx1416454/ID_code_bak/scripts/inference_mmdit_i2v_ip_last_no_cfg_jsonl_prompt.py", line 632, in main
    main()
  File "/data/huanan/code/jwx1416454/ID_code_bak/scripts/inference_mmdit_i2v_ip_last_no_cfg_jsonl_prompt.py", line 632, in main
    dist.barrier()
  File "/home/ma-user/anaconda3/envs/PyTorch-2.1.0/lib/python3.9/site-packages/torch/distributed/c10d_logger.py", line 47, in wrapper
    return func(*args, **kwargs)
  File "/home/ma-user/anaconda3/envs/PyTorch-2.1.0/lib/python3.9/site-packages/torch/distributed/distributed_c10d.py", line 3703, in barrier
        dist.barrier()dist.barrier()

  File "/home/ma-user/anaconda3/envs/PyTorch-2.1.0/lib/python3.9/site-packages/torch/distributed/c10d_logger.py", line 47, in wrapper
  File "/home/ma-user/anaconda3/envs/PyTorch-2.1.0/lib/python3.9/site-packages/torch/distributed/c10d_logger.py", line 47, in wrapper
        return func(*args, **kwargs)return func(*args, **kwargs)

  File "/home/ma-user/anaconda3/envs/PyTorch-2.1.0/lib/python3.9/site-packages/torch/distributed/distributed_c10d.py", line 3703, in barrier
  File "/home/ma-user/anaconda3/envs/PyTorch-2.1.0/lib/python3.9/site-packages/torch/distributed/distributed_c10d.py", line 3703, in barrier
    work.wait()
RuntimeError: The Inner error is reported as above. The process exits for this inner error, and the current working operator name is HcclAllreduce.
Since the operator is called asynchronously, the stacktrace may be inaccurate. If you want to get the accurate stacktrace, pleace set the environment variable ASCEND_LAUNCH_BLOCKING=1.
[ERROR] 2026-05-17-15:10:26 (PID:188122, Device:3, RankID:3) ERR00100 PTA call acl api failed
        work.wait()work.wait()

RuntimeErrorRuntimeError: : The Inner error is reported as above. The process exits for this inner error, and the current working operator name is HcclAllreduce.
Since the operator is called asynchronously, the stacktrace may be inaccurate. If you want to get the accurate stacktrace, pleace set the environment variable ASCEND_LAUNCH_BLOCKING=1.
[ERROR] 2026-05-17-15:10:26 (PID:188121, Device:2, RankID:2) ERR00100 PTA call acl api failedThe Inner error is reported as above. The process exits for this inner error, and the current working operator name is HcclAllreduce.
Since the operator is called asynchronously, the stacktrace may be inaccurate. If you want to get the accurate stacktrace, pleace set the environment variable ASCEND_LAUNCH_BLOCKING=1.
[ERROR] 2026-05-17-15:10:26 (PID:188123, Device:4, RankID:4) ERR00100 PTA call acl api failed

    dist.barrier()
  File "/home/ma-user/anaconda3/envs/PyTorch-2.1.0/lib/python3.9/site-packages/torch/distributed/c10d_logger.py", line 47, in wrapper
    return func(*args, **kwargs)
  File "/home/ma-user/anaconda3/envs/PyTorch-2.1.0/lib/python3.9/site-packages/torch/distributed/distributed_c10d.py", line 3703, in barrier
    main()
  File "/data/huanan/code/jwx1416454/ID_code_bak/scripts/inference_mmdit_i2v_ip_last_no_cfg_jsonl_prompt.py", line 632, in main
    work.wait()
RuntimeError: The Inner error is reported as above. The process exits for this inner error, and the current working operator name is HcclAllreduce.
Since the operator is called asynchronously, the stacktrace may be inaccurate. If you want to get the accurate stacktrace, pleace set the environment variable ASCEND_LAUNCH_BLOCKING=1.
[ERROR] 2026-05-17-15:10:26 (PID:188125, Device:6, RankID:6) ERR00100 PTA call acl api failed
        dist.barrier()dist.barrier()

  File "/home/ma-user/anaconda3/envs/PyTorch-2.1.0/lib/python3.9/site-packages/torch/distributed/c10d_logger.py", line 47, in wrapper
  File "/home/ma-user/anaconda3/envs/PyTorch-2.1.0/lib/python3.9/site-packages/torch/distributed/c10d_logger.py", line 47, in wrapper
        return func(*args, **kwargs)return func(*args, **kwargs)

  File "/home/ma-user/anaconda3/envs/PyTorch-2.1.0/lib/python3.9/site-packages/torch/distributed/distributed_c10d.py", line 3703, in barrier
  File "/home/ma-user/anaconda3/envs/PyTorch-2.1.0/lib/python3.9/site-packages/torch/distributed/distributed_c10d.py", line 3703, in barrier
    work.wait()
    work.wait()
RuntimeError: The Inner error is reported as above. The process exits for this inner error, and the current working operator name is HcclAllreduce.
Since the operator is called asynchronously, the stacktrace may be inaccurate. If you want to get the accurate stacktrace, pleace set the environment variable ASCEND_LAUNCH_BLOCKING=1.
[ERROR] 2026-05-17-15:10:26 (PID:188124, Device:5, RankID:5) ERR00100 PTA call acl api failed
RuntimeError: The Inner error is reported as above. The process exits for this inner error, and the current working operator name is HcclAllreduce.
Since the operator is called asynchronously, the stacktrace may be inaccurate. If you want to get the accurate stacktrace, pleace set the environment variable ASCEND_LAUNCH_BLOCKING=1.
[ERROR] 2026-05-17-15:10:26 (PID:188120, Device:1, RankID:1) ERR00100 PTA call acl api failed
    main()
  File "/data/huanan/code/jwx1416454/ID_code_bak/scripts/inference_mmdit_i2v_ip_last_no_cfg_jsonl_prompt.py", line 632, in main
    dist.barrier()
  File "/home/ma-user/anaconda3/envs/PyTorch-2.1.0/lib/python3.9/site-packages/torch/distributed/c10d_logger.py", line 47, in wrapper
    return func(*args, **kwargs)
  File "/home/ma-user/anaconda3/envs/PyTorch-2.1.0/lib/python3.9/site-packages/torch/distributed/distributed_c10d.py", line 3703, in barrier
    work.wait()
RuntimeError: The Inner error is reported as above. The process exits for this inner error, and the current working operator name is HcclAllreduce.
Since the operator is called asynchronously, the stacktrace may be inaccurate. If you want to get the accurate stacktrace, pleace set the environment variable ASCEND_LAUNCH_BLOCKING=1.
[ERROR] 2026-05-17-15:10:26 (PID:188131, Device:7, RankID:7) ERR00100 PTA call acl api failed
Saved attn map video to ./s3_id2_wo_ref2/I2V_ID_model.pth/id2/attn_maps/attn_sample00_t024.mp4
Saved attn map video to ./s3_id2_wo_ref2/I2V_ID_model.pth/id2/attn_maps/attn_sample00_t025.mp4
Saved attn map video to ./s3_id2_wo_ref2/I2V_ID_model.pth/id2/attn_maps/attn_sample00_t026.mp4
[2026-05-17 15:10:44,936] torch.distributed.elastic.multiprocessing.api: [WARNING] Sending process 188119 closing signal SIGTERM
[2026-05-17 15:10:47,905] torch.distributed.elastic.multiprocessing.api: [ERROR] failed (exitcode: 1) local_rank: 1 (pid: 188120) of binary: /home/ma-user/anaconda3/envs/PyTorch-2.1.0/bin/python3.9
Traceback (most recent call last):
  File "/home/ma-user/anaconda3/envs/PyTorch-2.1.0/bin/torchrun", line 8, in <module>
    sys.exit(main())
  File "/home/ma-user/anaconda3/envs/PyTorch-2.1.0/lib/python3.9/site-packages/torch/distributed/elastic/multiprocessing/errors/__init__.py", line 346, in wrapper
    return f(*args, **kwargs)
  File "/home/ma-user/anaconda3/envs/PyTorch-2.1.0/lib/python3.9/site-packages/torch/distributed/run.py", line 806, in main
    run(args)
  File "/home/ma-user/anaconda3/envs/PyTorch-2.1.0/lib/python3.9/site-packages/torch/distributed/run.py", line 797, in run
    elastic_launch(
  File "/home/ma-user/anaconda3/envs/PyTorch-2.1.0/lib/python3.9/site-packages/torch/distributed/launcher/api.py", line 134, in __call__
    return launch_agent(self._config, self._entrypoint, list(args))
  File "/home/ma-user/anaconda3/envs/PyTorch-2.1.0/lib/python3.9/site-packages/torch/distributed/launcher/api.py", line 264, in launch_agent
    raise ChildFailedError(
torch.distributed.elastic.multiprocessing.errors.ChildFailedError: 
============================================================
scripts/inference_mmdit_i2v_ip_last_no_cfg_jsonl_prompt.py FAILED
------------------------------------------------------------
Failures:
[1]:
  time      : 2026-05-17_15:10:44
  host      : notebook-030d6763-5eac-47ed-80c5-b6dc089cb3d4.notebook-030d6763-5eac-47ed-80c5-b6dc089cb3d4-distributed.default.svc.cluster.local
  rank      : 2 (local_rank: 2)
  exitcode  : 1 (pid: 188121)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
[2]:
  time      : 2026-05-17_15:10:44
  host      : notebook-030d6763-5eac-47ed-80c5-b6dc089cb3d4.notebook-030d6763-5eac-47ed-80c5-b6dc089cb3d4-distributed.default.svc.cluster.local
  rank      : 3 (local_rank: 3)
  exitcode  : 1 (pid: 188122)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
[3]:
  time      : 2026-05-17_15:10:44
  host      : notebook-030d6763-5eac-47ed-80c5-b6dc089cb3d4.notebook-030d6763-5eac-47ed-80c5-b6dc089cb3d4-distributed.default.svc.cluster.local
  rank      : 4 (local_rank: 4)
  exitcode  : 1 (pid: 188123)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
[4]:
  time      : 2026-05-17_15:10:44
  host      : notebook-030d6763-5eac-47ed-80c5-b6dc089cb3d4.notebook-030d6763-5eac-47ed-80c5-b6dc089cb3d4-distributed.default.svc.cluster.local
  rank      : 5 (local_rank: 5)
  exitcode  : 1 (pid: 188124)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
[5]:
  time      : 2026-05-17_15:10:44
  host      : notebook-030d6763-5eac-47ed-80c5-b6dc089cb3d4.notebook-030d6763-5eac-47ed-80c5-b6dc089cb3d4-distributed.default.svc.cluster.local
  rank      : 6 (local_rank: 6)
  exitcode  : 1 (pid: 188125)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
[6]:
  time      : 2026-05-17_15:10:44
  host      : notebook-030d6763-5eac-47ed-80c5-b6dc089cb3d4.notebook-030d6763-5eac-47ed-80c5-b6dc089cb3d4-distributed.default.svc.cluster.local
  rank      : 7 (local_rank: 7)
  exitcode  : 1 (pid: 188131)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
------------------------------------------------------------
Root Cause (first observed failure):
[0]:
  time      : 2026-05-17_15:10:44
  host      : notebook-030d6763-5eac-47ed-80c5-b6dc089cb3d4.notebook-030d6763-5eac-47ed-80c5-b6dc089cb3d4-distributed.default.svc.cluster.local
  rank      : 1 (local_rank: 1)
  exitcode  : 1 (pid: 188120)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
============================================================
[ERROR] TBE Subprocess[task_distribute] raise error[], main process disappeared!
[ERROR] TBE Subprocess[task_distribute] raise error[], main process disappeared!
[ERROR] TBE Subprocess[task_distribute] raise error[], main process disappeared!
[ERROR] TBE Subprocess[task_distribute] raise error[], main process disappeared!
[ERROR] TBE Subprocess[task_distribute] raise error[], main process disappeared!
[ERROR] TBE Subprocess[task_distribute] raise error[], main process disappeared!
[ERROR] TBE Subprocess[task_distribute] raise error[], main process disappeared!
[ERROR] TBE Subprocess[task_distribute] raise error[], main process disappeared!
[ma-user ID_code_bak]$/home/ma-user/anaconda3/envs/PyTorch-2.1.0/lib/python3.9/multiprocessing/resource_tracker.py:216: UserWarning: resource_tracker: There appear to be 30 leaked semaphore objects to clean up at shutdown
  warnings.warn('resource_tracker: There appear to be %d '
