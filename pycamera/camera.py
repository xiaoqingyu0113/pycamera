import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting 
import yaml
try:
    import torch
except ImportError:
    Warning("torch not imported")


def draw_camera(R, t, color='blue', scale=1, ax=None):

    '''
     Draw camera pose on matplotlib axis, for debugging or visualization

     R: cam_R_world
     t: world -> camera in camera coordinate
    '''
    points = np.array([[0,0,0],[1,1,2],[0,0,0],[-1,1,2],[0,0,0],[1,-1,2],[0,0,0],[-1,-1,2],
                        [-1,1,2],[1,1,2],[1,-1,2],[-1,-1,2]]) * scale

    x,y,z = R.T @ (points.T - t[:,np.newaxis])

    if ax is not None:
        ax.plot(x,y,z,color=color)
        return ax
    else:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.plot(x,y,z,color=color)
        return ax
    
def set_axes_equal(ax):
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    spans = np.abs(limits[:, 1] - limits[:, 0])
    max_span = max(spans)
    centers = np.mean(limits, axis=1)
    limits[:, 0] = centers - max_span / 2
    limits[:, 1] = centers + max_span / 2
    ax.set_xlim3d(limits[0])
    ax.set_ylim3d(limits[1])
    ax.set_zlim3d(limits[2])

def axis_equal(ax,X,Y,Z):
   # Set the limits of the axes to be equal

    x = np.array(X).flatten()
    y = np.array(Y).flatten()
    z = np.array(Z).flatten()

    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5
    ax.set_xlim3d(mid_x - max_range, mid_x + max_range)
    ax.set_ylim3d(mid_y - max_range, mid_y + max_range)
    ax.set_zlim3d(mid_z - max_range, mid_z + max_range)

    # Set labels for the axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')


class CameraParam:
    '''
    A camera paramter structure
    self.K  - intrinsics
    self.R  - cam_R_world
    self.t  - (world -> t_cam) in camera frame
    '''
    def __init__(self,K,R,t,distortion=None, filename=None, backend='numpy',device='cpu'):
        self.K = K 
        self.R = R 
        self.t = t
        self.d = distortion
        self.filename = filename
        self.backend = backend
        self.device = device

    def to_torch(self, device='cpu'):
        if self.backend == 'torch':
            return self
        self.K = torch.tensor(self.K, dtype=torch.float32).to(device)
        self.R = torch.tensor(self.R, dtype=torch.float32).to(device)
        self.t = torch.tensor(self.t, dtype=torch.float32).to(device)
        self.backend = 'torch'
        self.device = device
        return self

    def to_numpy(self):
        if self.backend == 'numpy':
            return self
        self.K = self.K.cpu().numpy()
        self.R = self.R.cpu().numpy()
        self.t = self.t.cpu().numpy()
        self.backend = 'numpy'
        self.device = 'cpu'
        return self

    def __repr__(self) -> str:
        return f"CameraParam: \n\tK: {self.K}\n\tR: {self.R}\n\tt: {self.t}\n\tdistortion: {self.d}\n\tfilename: {self.filename}"
    
    @classmethod
    def from_yaml(cls, yaml_file):
        def read_yaml_file(file_path):
            try:
                with open(file_path, 'r') as file:
                    # Load the YAML data into a Python dictionary
                    data = yaml.safe_load(file)
                return data
            except FileNotFoundError:
                print(f"File not found: {file_path}")
                return None
            except Exception as e:
                print(f"An error occurred while reading the YAML file: {e}")
                return None
            
        camera_param_raw = read_yaml_file(yaml_file)
        K  = np.array(camera_param_raw['camera_matrix']['data']).reshape(3,3)
        R =  np.array(camera_param_raw['rotation_matrix']['data']).reshape(3,3)
        t = np.array(camera_param_raw['translation'])
 
        camera_param = cls(K,R,t,distortion=None, filename=yaml_file, backend='numpy')
        return camera_param

    def parser(self):
        return self.K, self.R, self.t, self.d

    # def to_gtsam(self):
    #     K1 = self.K
    #     R1 = self.R
    #     t1 = -R1.T@self.t
    #     K1_gtsam = gtsam.Cal3_S2(K1[0,0], K1[1,1], K1[2,2], K1[0,2], K1[1,2])
    #     R1_gtsam = gtsam.Rot3(R1.T) 
    #     t1_gtsam = gtsam.Point3(t1[0],t1[1],t1[2])
    #     pose1 = gtsam.Pose3(R1_gtsam, t1_gtsam) # pose should be camera pose in the world frame
    #     return K1_gtsam, pose1        
    
    def proj2img(self,p,shape=None):
        if self.backend == 'numpy':
            return _proj2img_numpy(p, self.K, self.R, self.t)
        elif self.backend == 'torch':
            return _proj2img_torch(p, self.K, self.R, self.t, device=self.device)
        else:
            ValueError("backend in CameraPram not identified")

    
    def get_projection_matrix(self):
        t = np.array(self.t)
        t = np.expand_dims(t,axis = 1)
        R = np.array(self.R)
        K = np.array(self.K)
        M = K @ np.block([R,t])
        return M

    def draw(self,ax, color='blue',scale=1):
        draw_camera(self.R, self.t, color=color, scale=scale, ax=ax)
        return ax
    

def _proj2img_numpy(p, K, R, t):
    '''
        p - 3 or Nx3
        uv1 - 3 or Nx3
        uv - N x 2
    '''
    M = K @ np.block([R,t[:,np.newaxis]])
    if len(p.shape) ==1:
        uv_ =M@np.append(p,1)
        uv1 = uv_/uv_[2]
        uv = uv1[:2]
    else:
        N = p.shape[0]
        uv_ = M@np.append(p.T,np.ones((1,N)),axis=0)
        uv1 = uv_/uv_[2,:]
        uv = uv1.T[:,:2]

    return uv

def _proj2img_torch(p, K, R, t, device='cpu'):
    '''
    p - 3 or Nx3
    uv1 - 3 or Nx3
    uv - N x 2
    '''


    M = K @ torch.cat((R, t[:, None]), dim=-1)
    if len(p.shape) == 1:
        uv_ = M @ torch.cat((p, torch.tensor([1],dtype=torch.float32, device=device)))
        uv1 = uv_ / uv_[2]
        uv = uv1[:2]
    else:
        N = p.shape[0]
        uv_ = M @ torch.cat((p.T, torch.ones((1, N),dtype=torch.float32, device=device)), axis=0)
        uv1 = uv_ / uv_[2, :]
        uv = uv1.T[:, :2]
    return uv

def distort_pixel(uv, dist_coeffs, K):
    '''
    reference: https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
    '''
    cx = K[0,2]
    cy = K[1,2]
    fx = K[0,0]
    fy = K[1,1]
    
    k1, k2, p1, p2, k3 = dist_coeffs

    if len(uv.shape) == 1:
        x_d = (uv[0] - cx)/fx
        y_d = (uv[1] - cy)/fy
    else:
        x_d = (uv[:,0] - cx)/fx
        y_d = (uv[:,1] - cy)/fy

    
    r2 = x_d**2 + y_d**2

    # Radial distortion 
    x_u = x_d * (1 + k1 * r2 + k2 * r2**2 + k3 * r2**3)
    y_u = y_d * (1 + k1 * r2 + k2 * r2**2 + k3 * r2**3)

    # Tangential distortion 
    x_u += 2 * p1 * x_d * y_d + p2 * (r2 + 2 * x_d**2)
    y_u += p1 * (r2 + 2 * y_d**2) + 2 * p2 * x_d * y_d

    x_u = fx*x_u + cx
    y_u = fy*y_u + cy

    if len(uv.shape) == 1:
        rst = np.array([x_u, y_u])
    else:
        rst = np.c_[x_u, y_u]
        
    return rst

def undistort_pixel(uv, dist_coeffs, K, iter=2):
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    k1, k2, p1, p2, k3 = dist_coeffs[:5]

    # Initial guess: undistorted point is the same as the distorted point
    uv_undist = uv.copy()
    for _ in range(iter):  # Iterate 2 times or until convergence
        uv_dist = distort_pixel(uv_undist,dist_coeffs,K)
        uv_undist += (uv - uv_dist)
 
    return uv_undist


def world2camera(p,camera_params:CameraParam):
    '''
    p - 3 or Nx3
    uv1 - 3 or Nx3

    uv - N x 2
    '''
    K,R,t,d = camera_params.parser()
    if len(p.shape) ==1:
        uv_ = K@np.block([R,-R@t[:,np.newaxis]])@np.append(p,1)
        uv1 = uv_/uv_[2]
        return uv1[:2]
    else:
        N = p.shape[0]
        uv_ = K@np.block([R,-R@t[:,np.newaxis]])@np.append(p.T,np.ones((1,N)),axis=0)
        uv1 = uv_/uv_[2,:]

    uv = uv1.T[:,:2]
    return uv


def projection_img2world_line(uv,camera_params,z=-2.0):

    '''
        Solve for a projection line from image to realworld.

        input:
            uv - point on image
            camera_params - camera paramters
            z - end point of the projection line 
    '''

    M = camParam2proj(camera_params)
    
    u = uv[0];v = uv[1]
    A = np.array([
        [M[0,0] - u*M[2,0], M[0,1] - u*M[2,1]],
        [M[1,0] - v*M[2,0], M[1,1] - v*M[2,1]]
        ])
    b1 = np.array([
        [M[0,2] - u*M[2,2]],
        [M[1,2] - v*M[2,2]]
        ])
    b2 = np.array([
        [M[0,3] - u*M[2,3]],
        [M[1,3] - v*M[2,3]]
        ])
    xy = np.linalg.solve(A,-b1*z-b2)

    start = camera_params.t
    end = np.append(xy,z)

    return start, end

def plot_line_3d(ax,start, end, **kwargs):
    '''
        plot a line using starting point and end point 
    '''
    ax.plot([start[0],end[0]], [start[1],end[1]],[start[2],end[2]],**kwargs)


def camParam2proj(cam_param):
    '''
    computing for projection matrix
    '''
    t = np.array(cam_param.t)
    t = np.expand_dims(t,axis = 1)
    R = np.array(cam_param.R)
    K = np.array(cam_param.K)
    M = K @ np.block([R,-R@t])
    return M


def triangulate(uv_left: np.ndarray, uv_right: np.ndarray, left_camera_param:CameraParam, right_camera_param:CameraParam) -> np.ndarray:
    """
    Triangulate the 3D position of the ball given its 2D position in two stereo images.

    Returns:
        ball_position: 3D coordinates of the ball in world coordinates, np.ndarray (3,).
    """

    assert len(uv_left) == 2 
    assert len(uv_right) == 2
  

    # Convert points for triangulation
    # homogeneous_2d_points_left = cv2.undistortPoints(uv_left, left_camera_param.K, left_camera_param.d, None, left_camera_param.K)
    # homogeneous_2d_points_right = cv2.undistortPoints(uv_right, right_camera_param.K, right_camera_param.d, None, right_camera_param.K)
    homogeneous_2d_points_left = np.array(uv_left)
    homogeneous_2d_points_right = np.array(uv_right)

    # Triangulate
    points_4d_homogeneous = cv2.triangulatePoints(
        left_camera_param.get_projection_matrix(),  # Projection matrix for the left camera
        right_camera_param.get_projection_matrix(),    # Projection matrix for the right camera
        homogeneous_2d_points_left.T,
        homogeneous_2d_points_right.T
    )

    # Convert from homogeneous to 3D
    points_3d = points_4d_homogeneous / points_4d_homogeneous[3]
    ball_position = points_3d[:3].flatten()

    return ball_position

def closest_points_twoLine(p0,p1,q0,q1): # https://blog.csdn.net/Hunter_pcx/article/details/78577202

    '''
    solving for closest point of two line in 3d space 
    '''
    A = np.c_[p0-p1,q0-q1]
    b = q1-p1

    # solve normal equation
    x = np.linalg.solve(A.T@A, A.T@b)
    
    a = x[0]
    n = -x[1]

    pc = a*p0 + (1-a)*p1
    qc = n*q0 + (1-n)*q1

    return pc, qc
       


def pairwise_dist(x,y):
    """
    x: N,D
    y: M,D

    return N,M
    """

    d_sq = np.sum(x**2,axis=1)[:,np.newaxis] + np.sum(y**2,axis=1)[np.newaxis,:] - 2*x @ y.T

    d_sq[d_sq<0] = 0

    return np.sqrt(d_sq)

def rotm(th,axis):
    if axis == 'z':
        return np.array([[np.cos(th),-np.sin(th),0],
                        [np.sin(th),np.cos(th),0],
                        [0,0,1]]) 
    elif axis=='y':
        return np.array([[np.cos(th),0,np.sin(th)],
                        [0,1,0],
                        [-np.sin(th),0,np.cos(th)]])
    elif axis == 'x':
        return np.array([[1,0,0],
                            [0, np.cos(th),-np.sin(th)],
                            [0,np.sin(th),np.cos(th)]])
    else:
        ValueError("axis not identified")