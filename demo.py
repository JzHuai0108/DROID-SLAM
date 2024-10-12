import sys
sys.path.append('droid_slam')

from tqdm import tqdm
import numpy as np
import torch
import lietorch
import cv2
import os
import glob 
import time
import argparse
import yaml

from torch.multiprocessing import Process
from droid import Droid

import torch.nn.functional as F


def show_image(image):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(1)

def image_stream(imagedir, calib, stride):
    """ image generator """
    if 'rrxio' in imagedir:
        for t, image, intrinsics, stamp in rrxio_stream(imagedir, calib, stride):
            yield t, image, intrinsics, stamp
        return

    calib = np.loadtxt(calib, delimiter=" ")
    fx, fy, cx, cy = calib[:4]

    K = np.eye(3)
    K[0,0] = fx
    K[0,2] = cx
    K[1,1] = fy
    K[1,2] = cy

    image_list = sorted(os.listdir(imagedir))[::stride]

    for t, imfile in enumerate(image_list):
        image = cv2.imread(os.path.join(imagedir, imfile))
        if len(calib) > 4:
            image = cv2.undistort(image, K, calib[4:])

        h0, w0, _ = image.shape
        h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
        w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))

        image = cv2.resize(image, (w1, h1))
        image = image[:h1-h1%8, :w1-w1%8]
        image = torch.as_tensor(image).permute(2, 0, 1)

        intrinsics = torch.as_tensor([fx, fy, cx, cy])
        intrinsics[0::2] *= (w1 / w0)
        intrinsics[1::2] *= (h1 / h0)

        yield t, image[None], intrinsics, t


def rrxio_stream(imagedir, calib_yaml, stride, skip=0):
    image_list = sorted(os.listdir(imagedir))[::stride]

    with open(calib_yaml, 'r') as file:
        config = yaml.safe_load(file)
    raw_calib = config['Dataset']['Calibration']['raw']
    img_topic = config['Dataset']['img_topic']
    raw_K = np.eye(3)
    raw_K[0,0] = raw_calib['fx']
    raw_K[0,2] = raw_calib['cx']
    raw_K[1,1] = raw_calib['fy']
    raw_K[1,2] = raw_calib['cy']
    width = raw_calib['width']
    height = raw_calib['height']

    opt_calib = config['Dataset']['Calibration']['opt']
    opt_K = np.eye(3)
    opt_K[0,0] = opt_calib['fx']
    opt_K[0,2] = opt_calib['cx']
    opt_K[1,1] = opt_calib['fy']
    opt_K[1,2] = opt_calib['cy']

    if 'distortion_model' in raw_calib.keys():
        distortion_model = raw_calib['distortion_model']
    else:
        distortion_model = None
    print(f"Distortion model: {distortion_model}")

    if distortion_model == 'radtan':
        dist_coeffs = np.array(
            [
                raw_calib["k1"],
                raw_calib["k2"],
                raw_calib["p1"],
                raw_calib["p2"],
                raw_calib["k3"],
            ]
        )
        map1x, map1y = cv2.initUndistortRectifyMap(
            raw_K,
            dist_coeffs,
            np.eye(3),
            opt_K,
            (width, height),
            cv2.CV_32FC1,
        )
    elif distortion_model == 'equidistant':
        dist_coeffs = np.array(
            [
                raw_calib["k1"],
                raw_calib["k2"],
                raw_calib["k3"],
                raw_calib["k4"]
            ]
        )
        map1x, map1y = cv2.fisheye.initUndistortRectifyMap(
            raw_K,
            dist_coeffs,
            np.eye(3),
            opt_K,
            (width, height),
            cv2.CV_32FC1,
        )
    else:
        map1x, map1y = None, None

    for t, imfile in enumerate(image_list):
        image = cv2.imread(os.path.join(imagedir, imfile))
        if image.shape[2] == 1:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            rgb_image = image

        # Display the image (optional)
        cv2.imshow('Image', rgb_image)
        cv2.waitKey(1)  # Adjust the delay as needed (e.g., for video playback)

        time = float(imfile[:-4])
        if distortion_model is None:
            undistorted_image = rgb_image
        else:
            undistorted_image = cv2.remap(rgb_image, map1x, map1y, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)

        h0, w0, _ = undistorted_image.shape
        h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
        w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))

        undistorted_image = cv2.resize(undistorted_image, (w1, h1))
        undistorted_image = undistorted_image[:h1-h1%8, :w1-w1%8]
        undistorted_image = torch.as_tensor(undistorted_image).permute(2, 0, 1)

        intrinsics = torch.as_tensor([opt_calib['fx'], opt_calib['fy'], opt_calib['cx'], opt_calib['cy']])
        intrinsics[0::2] *= (w1 / w0)
        intrinsics[1::2] *= (h1 / h0)

        yield t, undistorted_image[None], intrinsics, time


def save_tum_traj(traj: np.ndarray, times: list, outfile: str):
    """
    Writes a trajectory and corresponding times to a TUM format file.

    Args:
        traj (np.ndarray): Nx7 array containing the trajectory data, where each row is [x, y, z, qx, qy, qz, qw].
        times (list): List of time stamps (Nx1) corresponding to the trajectory points.
        outfile (str): Output file path to save the trajectory data.

    Each line in the output file will have the format:
    time x y z qx qy qz qw
    """
    # Ensure that the trajectory and times have matching lengths
    if traj.shape[0] != len(times):
        raise ValueError("The length of times and the number of trajectory points must be the same.")

    # Open the file for writing
    with open(outfile, 'w') as f:
        for i in range(len(times)):
            # Extract time and trajectory data for the current step
            time = times[i]
            x, y, z, qx, qy, qz, qw = traj[i]

            # Write the time and trajectory in TUM format
            f.write(f"{time:.9f} {x:.6f} {y:.6f} {z:.6f} {qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}\n")


def save_reconstruction(droid, reconstruction_path):

    from pathlib import Path
    import random
    import string

    t = droid.video.counter.value
    tstamps = droid.video.tstamp[:t].cpu().numpy()
    images = droid.video.images[:t].cpu().numpy()
    disps = droid.video.disps_up[:t].cpu().numpy()
    poses = droid.video.poses[:t].cpu().numpy()
    intrinsics = droid.video.intrinsics[:t].cpu().numpy()

    Path("reconstructions/{}".format(reconstruction_path)).mkdir(parents=True, exist_ok=True)
    np.save("reconstructions/{}/tstamps.npy".format(reconstruction_path), tstamps)
    np.save("reconstructions/{}/images.npy".format(reconstruction_path), images)
    np.save("reconstructions/{}/disps.npy".format(reconstruction_path), disps)
    np.save("reconstructions/{}/poses.npy".format(reconstruction_path), poses)
    np.save("reconstructions/{}/intrinsics.npy".format(reconstruction_path), intrinsics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagedir", type=str, help="path to image directory")
    parser.add_argument("--calib", type=str, help="path to calibration file")
    parser.add_argument("--t0", default=0, type=int, help="starting frame")
    parser.add_argument("--stride", default=3, type=int, help="frame stride")

    parser.add_argument("--weights", default="droid.pth")
    parser.add_argument("--buffer", type=int, default=512)
    parser.add_argument("--image_size", default=[240, 320])
    parser.add_argument("--disable_vis", action="store_true")

    parser.add_argument("--beta", type=float, default=0.3, help="weight for translation / rotation components of flow")
    parser.add_argument("--filter_thresh", type=float, default=2.4, help="how much motion before considering new keyframe")
    parser.add_argument("--warmup", type=int, default=8, help="number of warmup frames")
    parser.add_argument("--keyframe_thresh", type=float, default=4.0, help="threshold to create a new keyframe")
    parser.add_argument("--frontend_thresh", type=float, default=16.0, help="add edges between frames whithin this distance")
    parser.add_argument("--frontend_window", type=int, default=25, help="frontend optimization window")
    parser.add_argument("--frontend_radius", type=int, default=2, help="force edges between frames within radius")
    parser.add_argument("--frontend_nms", type=int, default=1, help="non-maximal supression of edges")

    parser.add_argument("--backend_thresh", type=float, default=22.0)
    parser.add_argument("--backend_radius", type=int, default=2)
    parser.add_argument("--backend_nms", type=int, default=3)
    parser.add_argument("--upsample", action="store_true")
    parser.add_argument("--reconstruction_path", help="path to saved reconstruction")
    parser.add_argument("--traj_path", help="path to saved trajectory")
    args = parser.parse_args()

    args.stereo = False
    torch.multiprocessing.set_start_method('spawn')

    droid = None

    # need high resolution depths
    if args.reconstruction_path is not None:
        args.upsample = True

    tstamps = []
    for (t, image, intrinsics, stamp) in tqdm(image_stream(args.imagedir, args.calib, args.stride)):
        if t < args.t0:
            continue

        if not args.disable_vis:
            show_image(image[0])

        if droid is None:
            args.image_size = [image.shape[2], image.shape[3]]
            droid = Droid(args)
        
        droid.track(t, image, intrinsics=intrinsics)

    if args.reconstruction_path is not None:
        save_reconstruction(droid, args.reconstruction_path)

    se3_traj, frame_stamps = droid.traj_filler(image_stream(args.imagedir, args.calib, args.stride))
    traj_est = se3_traj.inv().data.cpu().numpy()
    traj_est_lc, frame_stamps = droid.terminate(image_stream(args.imagedir, args.calib, args.stride))
    if args.traj_path is not None:
        save_tum_traj(traj_est, frame_stamps, args.traj_path)
        traj_path_lc = args.traj_path[:-4] + '_lc' + args.traj_path[-4:]
        save_tum_traj(traj_est_lc, frame_stamps, traj_path_lc)
    print('Done')
