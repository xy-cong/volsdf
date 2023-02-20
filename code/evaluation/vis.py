from wis3d import Wis3D
cameras_dir = "/mnt/data/cxy_colmap/volsdf/evals/cameras"
wis3d = Wis3D(cameras_dir, "test")
wis3d.add_spheres(np.array([0, 0, 0]), 3, name='sphere0')
# wis3d.add_spheres(np.array([[0, 1, 0], [0, 0, 1]]), 0.5, name = 'sphere1')
# wis3d.add_spheres(np.array([[0, 1, 0], [0, 0, 1]]), np.array([0.25, 0.5]),np.array([[0, 255, 0], [0, 0, 255]]), name='sphere2')
print(len(eval_dataloader))
for data_index, (indices, model_input, ground_truth) in enumerate(eval_dataloader):
    T = model_input["pose"].reshape(4,4)
    # T = pose_cvt(T)
    # print(T)
    # T_inv = np.linalg.inv(T)
    T_inv = T
    wis3d.add_camera_trajectory(T_inv[None], name='poses{:0>3d}_'.format(data_index))