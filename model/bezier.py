class BezierTorch_3:
    def __init__(self, control_points):
        # control_points 的形状应为 (batch_size, 4, dim)
        self.control_points = control_points
    
    def __call__(self, t):
        # t 的形状应为 (batch_size, num_points)
        t = t.unsqueeze(-1)  # 转换为 (batch_size, num_points, 1)
        
        # 三次贝塞尔曲线公式
        term0 = (1 - t)**3 * self.control_points[:, 0].unsqueeze(1)
        term1 = 3 * (1 - t)**2 * t * self.control_points[:, 1].unsqueeze(1)
        term2 = 3 * (1 - t) * t**2 * self.control_points[:, 2].unsqueeze(1)
        term3 = t**3 * self.control_points[:, 3].unsqueeze(1)
        
        # 返回形状为 (batch_size, num_points, dim)
        return term0 + term1 + term2 + term3
    
class BezierTorch_2:
    def __init__(self, control_points):
        # control_points 的形状应为 (batch_size, 3, dim)
        self.control_points = control_points
    
    def __call__(self, t):
        # t 的形状应为 (batch_size, num_points)
        t = t.unsqueeze(-1)  # 转换为 (batch_size, num_points, 1)
        
        # 二次贝塞尔曲线公式
        term0 = (1 - t)**2 * self.control_points[:, 0].unsqueeze(1)
        term1 = 2 * (1 - t) * t * self.control_points[:, 1].unsqueeze(1)
        term2 = t**2 * self.control_points[:, 2].unsqueeze(1)
        
        # 返回形状为 (batch_size, num_points, dim)
        return term0 + term1 + term2

