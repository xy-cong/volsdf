import cv2
import numpy as np
import os
import os.path as osp
from loguru import logger
from wis3d.wis3d import Wis3D
from src.colmap.read_write_model import qvec2rotmat
from src.utils.colmap.read_write_model import read_model
from src.post_optimization.utils.geometry_utils import convert_pose2T


def make_matching_plot_fast(image0, image1, kpts0, kpts1,
                            mkpts0, mkpts1, color, text,
                            margin=10, show_keypoints=True, num_matches_to_show=None):
    """draw matches in two images"""
    H0, W0 = image0.shape
    H1, W1 = image1.shape
    H, W = max(H0, H1), W0 + W1 + margin
    
    out = 255 * np.ones((H, W), np.uint8)
    out[:H0, :W0] = image0
    out[:H1, W0+margin:] = image1
    out = np.stack([out] * 3, -1)
    
    if show_keypoints:
        kpts0, kpts1 = np.round(kpts0).astype(int), np.round(kpts1).astype(int)
        white = (255, 255, 255)
        black = (0, 0, 0)
        
        for x, y in kpts0:
            cv2.circle(out, (x, y), 2, black, -1, lineType=cv2.LINE_AA)
            cv2.circle(out, (x, y), 1, white, -1, lineType=cv2.LINE_AA)
        for x, y in kpts1:
            cv2.circle(out, (x + margin + W0, y), 2, black, -1, lineType=cv2.LINE_AA)
            cv2.circle(out, (x + margin + W0, y), 1, white, -1, lineType=cv2.LINE_AA)

    mkpts0, mkpts1 = np.round(mkpts0).astype(int), np.round(mkpts1).astype(int)
    color = (np.array(color[:, :3]) * 255).astype(int)[:, ::-1]
    for (x0, y0), (x1, y1), c in zip(mkpts0, mkpts1, color):
        c = c.tolist()
        cv2.line(out, (x0, y0), (x1 + margin + W0, y1), 
                 color=c, thickness=1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x0, y0), 2, c, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x1 + margin + W0, y1), 2, c, -1,
                   lineType=cv2.LINE_AA)
    
    # FIXME: vis num_matches_to_show pairs
    if num_matches_to_show:
        pass

    scale = min(H / 640., 2.0) * 1.2 # scale
    Ht = int(30 * scale) # text height
    text_color_fg = (0, 225, 255)
    text_color_bg = (0, 0, 0)
    for i, t in enumerate(text):
        cv2.putText(out, t, (int(8*scale), Ht*(i+1)), cv2.FONT_HERSHEY_DUPLEX,
                    1.0*scale, text_color_bg, 3, cv2.LINE_AA)
        cv2.putText(out, t, (int(8*scale), Ht*(i+1)), cv2.FONT_HERSHEY_DUPLEX,
                    1.0*scale, text_color_fg, 1, cv2.LINE_AA)
        
    cv2.namedWindow('vis', 0)
    cv2.resizeWindow('vis', 800, 800)
    cv2.imshow('vis', out)
    cv2.waitKey(0)


def vis_match_pairs(pred, feats0, feats1, name0, name1):
    """vis matches on two images"""
    import matplotlib.cm as cm

    image0_path = name0
    image1_path = name1
    
    image0 = cv2.imread(image0_path)
    image0 = cv2.cvtColor(image0, cv2.COLOR_RGB2GRAY)
    image1 = cv2.imread(image1_path)
    image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
    
    matches = pred['matches0'][0].detach().cpu().numpy()
    valid = matches > -1
    
    kpts0, kpts1 = feats0['keypoints'].__array__(), feats1['keypoints'].__array__()
    mkpts0, mkpts1 = kpts0[valid], kpts1[matches[valid]]
    
    conf = pred['matching_scores0'][0].detach().cpu().numpy()
    mconf = conf[valid]
    color = cm.jet(mconf)

    make_matching_plot_fast(
        image0, image1, kpts0, kpts1,
        mkpts0, mkpts1, color, text=[]
    )


def reproj(K, pose, pts_3d):
    """ 
    Reproj 3d points to 2d points 
    @param K: [3, 3] or [3, 4]
    @param pose: [3, 4] or [4, 4]
    @param pts_3d: [n, 3]
    """
    assert K.shape == (3, 3) or K.shape == (3, 4)
    assert pose.shape == (3, 4) or pose.shape == (4, 4)

    if K.shape == (3, 3):
        K_homo = np.concatenate([K, np.zeros((3, 1))], axis=1)
    else:
        K_homo = K
    
    if pose.shape == (3, 4):
        pose_homo = np.concatenate([pose, np.array([[0, 0, 0, 1]])], axis=0)
    else:
        pose_homo = pose
    
    pts_3d = pts_3d.reshape(-1, 3)
    pts_3d_homo = np.concatenate([pts_3d, np.ones((pts_3d.shape[0], 1))], axis=1)
    pts_3d_homo = pts_3d_homo.T

    reproj_points = K_homo @ pose_homo @ pts_3d_homo
    reproj_points = reproj_points[:] / reproj_points[2:]
    reproj_points = reproj_points[:2, :].T
    return reproj_points # [n, 2]


def ransac_PnP(K, pts_2d, pts_3d, scale=1):
    """ solve pnp """
    dist_coeffs = np.zeros(shape=[8, 1], dtype='float64')
    
    pts_2d = np.ascontiguousarray(pts_2d.astype(np.float64))
    pts_3d = np.ascontiguousarray(pts_3d.astype(np.float64)) 
    K = K.astype(np.float64)
    
    pts_3d *= scale
    try:
        _, rvec, tvec, inliers = cv2.solvePnPRansac(pts_3d, pts_2d, K, dist_coeffs, reprojectionError=5,
                                                    iterationsCount=10000, flags=cv2.SOLVEPNP_EPNP)
        # _, rvec, tvec, inliers = cv2.solvePnPRansac(pts_3d, pts_2d, K, dist_coeffs)

        rotation = cv2.Rodrigues(rvec)[0]

        tvec /= scale
        pose = np.concatenate([rotation, tvec], axis=-1)
        pose_homo = np.concatenate([pose, np.array([[0, 0, 0, 1]])], axis=0)

        return pose, pose_homo, inliers
    except cv2.error:
        print("CV ERROR")
        return np.eye(4)[:3], np.eye(4), []


def draw_3d_box(image, corners_2d, linewidth=3, color='g'):
    """ Draw 3d box corners 
    @param corners_2d: [8, 2]
    """
    lines = np.array([
        [0, 1, 5, 4, 2, 3, 7, 6, 0, 1, 5, 4],
        [1, 5, 4, 0, 3, 7, 6, 2, 2, 3, 7, 6]
    ]).T

    colors = {
        'g': (0, 255, 0),
        'r': (0, 0, 255),
        'b': (255, 0, 0)
    }
    if color not in colors.keys():
        color = (42, 97, 247)
    else:
        color = colors[color]
    
    for id, line in enumerate(lines):
        pt1 = corners_2d[line[0]].astype(int)
        pt2 = corners_2d[line[1]].astype(int)
        cv2.line(image, tuple(pt1), tuple(pt2), color, linewidth)


def draw_2d_box(image, corners_2d, linewidth=3):
    """ Draw 2d box corners
    @param corners_2d: [x_left, y_top, x_right, y_bottom]
    """
    x1, y1, x2, y2 = corners_2d.astype(int)
    box_pts = [
        [(x1, y1), (x1, y2)],
        [(x1, y2), (x2, y2)],
        [(x2, y2), (x2, y1)],
        [(x2, y1), (x1, y1)]
    ]

    for pts in box_pts:
        pt1, pt2 = pts
        cv2.line(image, pt1, pt2, (0, 0, 255), linewidth)

def save_colmap_ws_to_vis3d(colmap_dir, save_path, name_prefix=''):
    if not osp.exists(colmap_dir):
        logger.warning(f"{colmap_dir} not exists!")
        return
    cameras, images, points3D = read_model(colmap_dir)
    save_path, name = save_path.rsplit('/',1)
    wis3d = Wis3D(save_path, name)

    # Point cloud:
    coord3D = []
    color = []
    # filter_track_length_thr = 4
    filter_track_length_thr = None
    if filter_track_length_thr is not None:
        logger.info(f"Output point cloud with track length thr: {filter_track_length_thr}")
    for point3D in points3D.values():
        if filter_track_length_thr is not None and len(point3D.image_ids) <= filter_track_length_thr:
            continue
        coord3D.append(point3D.xyz)
        color.append(point3D.rgb)
    if len(coord3D) == 0:
        logger.warning(f"Empty point cloud in {colmap_dir}")
    else:
        coord3D = np.stack(coord3D)
        color = np.stack(color)
        wis3d.add_point_cloud(coord3D, color, name= f'point_cloud_{name_prefix}')

    # Camera tragetory:
    for id, image in images.items():
        R = qvec2rotmat(image.qvec)
        t = image.tvec
        T = convert_pose2T([R, t])
        T = pose_cvt(T)
        T_inv = np.linalg.inv(T)
        wis3d.add_camera_trajectory(T_inv[None], name='poses{:0>3d}_'.format(id) + name_prefix)

def pose_cvt(T_cv, scale=1):
    """
    Convert transformation from CV representation
    Input: 4x4 Transformation matrix
    Output: 4x4 Converted transformation matrix
    """
    R = T_cv[:3, :3]
    t = T_cv[:3, 3]

    R_rot = np.eye(3)
    R_rot[1, 1] = -1
    R_rot[2, 2] = -1

    R = np.matmul(R_rot, R)
    t = np.matmul(R_rot, t)

    t *= scale

    T_cg = np.eye(4)
    T_cg[:3, :3] = R
    T_cg[:3, 3] = t
    return T_cg