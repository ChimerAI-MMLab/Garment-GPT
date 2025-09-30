import sys
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Union, List, Dict, Any
from tqdm import tqdm


class RTDataset(Dataset):
    def __init__(self, pattern_jsons_dirs: Union[str, List[str]], gcd_split_json: Union[str, List[str]], gcd_root_dir: Union[str, List[str]], split_type: Union[str, List[str]]):
        super().__init__()
        if isinstance(pattern_jsons_dirs, str):
            pattern_jsons_dirs = [pattern_jsons_dirs]
        self.translations = []
        self.rotations = []
        self.location = []
        # 处理多目录输入
        json_files = []
        print("scanning datasets...")
        
        # 先处理非GarmentCodeData数据集
        if pattern_jsons_dirs:
            for dir_path in pattern_jsons_dirs:
                if not os.path.isdir(dir_path):
                    raise ValueError(f"Directory {dir_path} does not exist")

                for root, dirs, files in os.walk(dir_path):
                    # 跳过GarmentCodeData目录，它将通过split_file单独处理
                    if "GarmentCodeData" in root or "GCD" in root:
                        continue
                        
                    for file in files:
                        if file.endswith("specification.json") or (("卫衣" in file) and file.endswith(".json")):
                            if "SewFactory" in root or "zenodo" in root or "own_data" in root:
                                json_files.append(os.path.join(root, file))
            
        # 处理GarmentCodeData数据集（通过split_file）
        if isinstance(gcd_split_json, str):
            gcd_split_json = [gcd_split_json]
        if isinstance(gcd_root_dir, str):
            gcd_root_dir = [gcd_root_dir]
        if isinstance(split_type, str):
            split_type = [split_type]
        else:
            split_type = list(split_type)
        # 处理GarmentCodeData数据集（通过split_file和gcd_root_dir列表依次加载）
        if gcd_split_json:
            for idx, split_f in enumerate(gcd_split_json):
                if not os.path.exists(split_f):
                    continue
                with open(split_f, "r") as f:
                    split_data = json.load(f)
                # 直接遍历split_type
                split_type_map = {
                    'train': 'training',
                    'test': 'test',
                    'val': 'validation',
                    'all': 'all'
                }
                for st in split_type:
                    if st in split_type_map:
                        split_key = split_type_map[st]
                    else:
                        split_key = st
                    split_paths = split_data.get(split_key, [])
                    if len(gcd_root_dir) > idx and gcd_root_dir[idx] is not None:
                        cur_gcd_root = gcd_root_dir[idx]
                    else:
                        cur_gcd_root = os.path.dirname(split_f)
                    for path in split_paths:
                        last_dir = os.path.basename(path)
                        full_path = os.path.join(cur_gcd_root, path, f"{last_dir}_specification.json")
                        if os.path.exists(full_path):
                            # 处理所有找到的JSON文件
                            json_files.append(full_path)
        for file_path in tqdm(json_files):
            # if ("GarmentCodeData" in file_path) and "specification.json" in file_path:
            if ("GarmentCodeData" in file_path or "GCD" in file_path) and "specification.json" in file_path:
                self._process_specification(file_path)
            elif "SewFactory" in file_path and "specification.json" in file_path:
                self._process_specification(file_path)
            elif "zenodo" in file_path and "specification.json" in file_path:
                self._process_specification(file_path)
            elif "own_data" in file_path and "specification.json" in file_path:
                self._process_specification(file_path)
            

    def _euler_to_quaternion(self, euler_degrees: List[float]) -> List[float]:
        """
        将欧拉角（度数）转换为四元数，假设顺序为ZYX（Yaw-Pitch-Roll）
        """
        yaw = np.radians(euler_degrees[2])
        pitch = np.radians(euler_degrees[1])
        roll = np.radians(euler_degrees[0])

        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy

        return [w, x, y, z]

    def _process_specification(self, file_path: str):
        with open(file_path, "r") as f:
            data = json.load(f)

            dir_path = os.path.dirname(file_path)

            panels = data.get("pattern", {}).get("panels", {})
            for panel_name, panel_data in panels.items():
                translation = panel_data["translation"]
                rotation_euler = panel_data["rotation"]

                rotation_quat = self._euler_to_quaternion(rotation_euler)

                self.translations.append(torch.tensor(translation, dtype=torch.float32))
                self.rotations.append(torch.tensor(rotation_quat, dtype=torch.float32))
                self.location.append(f"{dir_path}/panels/{panel_name}/rt")

    def __len__(self):
        return len(self.translations)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        translation = self.translations[idx]
        rotation = self.rotations[idx]
        location = self.location[idx]
        rt = torch.cat([translation, rotation])
        return {"translation": translation, "rotation": rotation, "rt": rt, "location": location}


if __name__ == "__main__":
    dataset_dir = ["/data/codec_data/zenodo/train", "/data/codec_data/sewfactory"]
    dataset = RTDataset(dataset_dir)
    print(len(dataset))
    print(dataset[0])
