import fire
import json
import os
import re
import numpy as np
import torch
import yaml
from PIL import Image
from typing import List, Optional, Dict
from vllm import LLM, SamplingParams

from llamafactory.hparams import get_infer_args
from llamafactory.model import load_tokenizer
from llamafactory.data import get_template_and_fix_tokenizer
# from llamafactory.data.constants import IMAGE_TOKEN

from model.codec_raw_allcubic import VQAutoCodec
from model.codec_qkv import VQTransformer
from train_rt.model import VQRTCodec
from transformers import AutoProcessor
# except ImportError as e:
#     print("="*80)
#     print("ERROR: Could not import custom model classes (VQAutoCodec, VQRTCodec, etc.).")
#     print("Please make sure the root directory of your 'codec_codebook' project is in your PYTHONPATH.")
#     print(f"Original error: {e}")
#     print("="*80)
#     exit(1)

import logging

logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def read_yaml_config(path: str) -> Dict:
    """读取 YAML 配置文件。"""
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def parse_garment_spidx_pure(text_list, num_q):
    garments = []
    for idx, s in enumerate(text_list): # 添加 enumerate 以便追踪索引
        try:
            if s == None:
                garments.append(None)
                continue
            s = s.split("<|endoftext|>", 1)[0] # 使用 split(..., 1) 更安全
            
            garment = None # 初始化为 None

            if "<SoS>" not in s:
                # --- 无缝合线逻辑 ---
                logger.info(f"记录 {idx} 不包含缝合线标记 <SoS>")
                # 修改正则表达式以匹配并捕获 <SoG> 后面的数字
                garment_pattern = re.compile(r"<SoG>(\d*)(.*?)<EoG>", re.DOTALL)
                match = garment_pattern.search(s) # 使用 search 可能更鲁棒
                if not match:
                    # 如果连 SoG...EoG 都匹配不上，直接跳过这个记录
                    logger.warning(f"跳过记录 {idx}: 无法找到 <SoG>...<EoG> 模式。")
                    garments.append(None)
                    continue

                scale_str, panels_str = match.groups()
                # 将捕获的 scale 转换为整数（如果存在）
                scale = int(scale_str) if scale_str else -1
                
                garment = {"name": "None", "scale": scale, "panels": []}

                panel_pattern = re.compile(r"<SoP>(.*?)<EoP>", re.DOTALL)
                panels = panel_pattern.findall(panels_str)
                if not panels:
                    logger.warning(f"跳过记录 {idx}: <SoG>...<EoG> 内没有找到面板。")
                    garments.append(None)
                    continue
                
                valid_panel_found = False
                for panel in panels:
                    # 修改面板模式以处理面板名称和边缘
                    p_match = re.search(r"(.*?)(?:<SoE>(.*?)<EoE>)?", panel, re.DOTALL)
                    if p_match:
                        p_name, edges_str = p_match.groups()
                        if not p_name:
                             logger.warning(f"警告: 跳过记录 {idx} 中的面板，因为缺少名称。")
                             continue
                        
                        edges = []
                        if edges_str: # 只有当 edges_str 存在时才解析
                            edge_indices = re.findall(r"<\|EID_(\d+)\|>", edges_str)
                            if edge_indices:
                                edge_indices = [int(index) for index in edge_indices]
                                n = len(edge_indices) // num_q
                                if len(edge_indices) % num_q == 0 and n > 0:
                                    edges = np.array(edge_indices).reshape(n, num_q).tolist()
                                else:
                                    logger.warning(f"警告: 记录 {idx}, 面板 '{p_name}' 中边缘数据格式不正确。跳过边缘。")
                        garment["panels"].append({"name": p_name.strip(), "edges": edges})
                        valid_panel_found = True
                    else:
                         logger.warning(f"警告: 跳过记录 {idx} 中的面板，因为模式不匹配。")
                
                if not valid_panel_found:
                    logger.warning(f"跳过记录 {idx}: 无法解析有效面板。")
                    garment = None # 重置 garment 如果没有有效 panel

            else:
                # --- 有缝合线逻辑 ---
                logger.info(f"记录 {idx} 包含缝合线标记 <SoS>")
                # 先记录原始缝合线文本用于调试
                sos_index = s.find("<SoS>")
                eos_index = s.find("<EoS>", sos_index)
                if sos_index >= 0 and eos_index > sos_index:
                    raw_stitches = s[sos_index+5:eos_index]
                    logger.info(f"记录 {idx} 原始缝合线文本: {raw_stitches[:100]}..." if len(raw_stitches) > 100 else raw_stitches)
                
                # 修改正则表达式以匹配并捕获 <SoG> 后面的数字
                garment_pattern = re.compile(r"<SoG>(\d*)(.*?)<SoS>(.*?)<EoS><EoG>", re.DOTALL)
                match = garment_pattern.search(s) # 使用 search
                if not match:
                    logger.warning(f"跳过记录 {idx}: 无法找到 <SoG>...<SoS>...<EoS><EoG> 模式。")
                    garments.append(None)
                    continue

                scale_str, panels_str, stitches_str = match.groups()
                # 将捕获的 scale 转换为整数（如果存在）
                scale = int(scale_str) if scale_str else -1
                
                garment = {"name": "None", "scale": scale, "panels": [], "stitches": []}

                panel_pattern = re.compile(r"<SoP>(.*?)<EoP>", re.DOTALL)
                panels = panel_pattern.findall(panels_str)
                if not panels:
                    logger.warning(f"跳过记录 {idx}: <SoG>...<EoG> 内没有找到面板 (带缝合线)。")
                    garments.append(None)
                    continue

                valid_panel_found = False
                for panel in panels:
                    p_match = re.search(r"(.*?)(?:<SoL>(.*?)<EoL>)?<SoE>(.*?)<EoE>", panel, re.DOTALL)
                    if p_match:
                        p_name, loc_str, edges_str = p_match.groups()
                        if not p_name or not edges_str: # loc_str 可以是 None
                            logger.warning(f"警告: 跳过记录 {idx} 中的面板，因为缺少名称或边缘数据。")
                            continue
                        
                        location = []
                        if loc_str:
                            loc_indices = re.findall(r"<\|LID_(\d+)\|>", loc_str)
                            if loc_indices:
                                location = [int(index) for index in loc_indices]
                            else:
                                logger.warning(f"警告: 记录 {idx}, 面板 '{p_name}' 中位置数据格式不正确。跳过位置。")

                        edges = []
                        edge_indices = re.findall(r"<\|EID_(\d+)\|>", edges_str)
                        if edge_indices:
                            edge_indices = [int(index) for index in edge_indices]
                            n = len(edge_indices) // num_q
                            if len(edge_indices) % num_q == 0 and n > 0:
                                edges = np.array(edge_indices).reshape(n, num_q).tolist()
                            else:
                                logger.warning(f"警告: 记录 {idx}, 面板 '{p_name}' 中边缘数据格式不正确。跳过边缘。")
                        
                        garment["panels"].append({"name": p_name.strip(), "location": location, "edges": edges})
                        valid_panel_found = True
                    else:
                        logger.warning(f"警告: 跳过记录 {idx} 中的面板，因为模式不匹配。")
                
                if not valid_panel_found:
                     logger.warning(f"跳过记录 {idx}: 无法解析有效面板 (带缝合线)。")
                     garment = None # 重置 garment 如果没有有效 panel
                     garments.append(None)
                     continue # 跳到下一个记录

                # 解析 Stitches
                if stitches_str:
                    logger.info(f"记录 {idx} 缝合线文本: {stitches_str[:100]}..." if len(stitches_str) > 100 else stitches_str)
                    
                    # 添加专门针对[panel:edge,panel:edge]格式的解析
                    stitch_pairs = re.findall(r'\[([\w_]+):(\d+),([\w_]+):(\d+)\]', stitches_str)
                    if stitch_pairs:
                        logger.info(f"找到 {len(stitch_pairs)} 个[panel:edge,panel:edge]格式的缝合线对")
                        result = []
                        for panel1, edge1, panel2, edge2 in stitch_pairs:
                            obj = {
                                "panel1": panel1,
                                "edge1": int(edge1),
                                "panel2": panel2, 
                                "edge2": int(edge2)
                            }
                            result.append(obj)
                        garment["stitches"] = result
                        logger.info(f"成功解析 {len(result)} 个缝合线记录")
                        garments.append(garment)
                        continue
                    
                    # 如果上面的方法不成功，尝试其他格式解析方法
                    # 1. 标准JSON格式 - 尝试直接解析
                    try:
                        # 先尝试修复可能的JSON格式问题
                        fixed_json = stitches_str.replace("'", '"').replace('""', '"')
                        # 然后尝试作为JSON数组解析
                        if fixed_json.strip().startswith('[') and fixed_json.strip().endswith(']'):
                            result = json.loads(fixed_json)
                            if isinstance(result, list):
                                garment["stitches"] = result
                                logger.info(f"成功解析记录 {idx} 中的缝合线为JSON数组")
                                garments.append(garment)
                                continue
                    except json.JSONDecodeError:
                        # 如果不是有效的JSON，继续尝试其他格式
                        logger.info(f"记录 {idx} 的缝合线不是有效的JSON数组，尝试其他解析方法")
                    
                    # 2. 使用原有的块解析方法
                    blocks = re.findall(r"\{(.*?)\}", stitches_str) # 匹配 {} 包裹的内容
                    if blocks:
                        logger.info(f"找到 {len(blocks)} 个缝合线块")
                    else:
                        logger.warning(f"没有找到缝合线块，使用更宽松的方法尝试解析")
                        # 尝试更宽松的匹配
                        blocks = re.findall(r'([^{}]+?(?:panel[^{}]*?\d+[^{}]*?panel[^{}]*?\d+))', stitches_str)
                    
                    result = []
                    for block in blocks:
                        try:
                            # 分割键值对并解析为字典
                            obj = {}
                            # 尝试多种键值对格式
                            pairs = re.findall(r'"(.*?)":\s*(\d+)', block) # 标准格式 "key": num
                            if not pairs:
                                pairs = re.findall(r"'(.*?)':\s*(\d+)", block) # 单引号格式 'key': num
                            if not pairs:
                                pairs = re.findall(r"(\w+):\s*(\d+)", block) # 无引号格式 key: num
                            if not pairs:
                                # 尝试直接提取panel索引
                                panel_pairs = re.findall(r"panel\s*(\d+).*?panel\s*(\d+)", block, re.DOTALL | re.IGNORECASE)
                                if panel_pairs:
                                    for p1, p2 in panel_pairs:
                                        obj["panel1"] = int(p1)
                                        obj["panel2"] = int(p2)
                            
                            if pairs:
                                for L, R in pairs:
                                    obj[L] = int(R)
                                    
                            if obj:
                                result.append(obj)
                            else:
                                logger.warning(f"警告: 无法从块中提取键值对: '{block}'")
                        except Exception as stitch_err:
                             logger.error(f"警告: 解析记录 {idx} 中的缝合线块时出错: '{block}'. 错误: {stitch_err}. 跳过块。")
                    
                    if result:
                        garment["stitches"] = result
                        logger.info(f"成功解析 {len(result)} 个缝合线记录")
                    else:
                        # 保留原始缝合线文本以备后续处理
                        garment["raw_stitches"] = stitches_str
                        logger.warning(f"无法解析缝合线为结构化数据，保留原始文本")
                else:
                    logger.warning(f"警告: 记录 {idx} 中预期的缝合线块缺失。")

            garments.append(garment)

        except Exception as e:
            logger.error(f"处理记录 {idx} 时出错: {e}. 跳过记录。", exc_info=True)
            # 发生任何未捕获的异常时，添加 None
            garments.append(None)
            continue # 继续处理下一个记录

    return garments

class GarmentPredictor:
    def __init__(
        self,
        llm_model_path: str,
        codec_config_path: str,
        rt_config_path: str,
        device: str = "cuda:0",
        template: str = "llava",
    ):
        self.device = torch.device(device)
        self.template_name = template
        self.num_q = 5 # 从你的 result_process.py 中硬编码

        print("--- Initializing Models ---")
        self._init_llm(llm_model_path)
        self._init_codec_models(codec_config_path, rt_config_path)
        print("--- All models initialized successfully ---")

    def _init_llm(self, llm_model_path: str):
        """加载并初始化 vLLM 引擎。"""
        print(f"1/3: Loading LLM from: {llm_model_path}")
        self.model_args, self.data_args, _, self.generating_args = get_infer_args(
            dict(
                model_name_or_path=llm_model_path,
                template=self.template_name,
                cutoff_len=4096, max_new_tokens=4096, temperature=0.1,
                vllm_maxlen=16384,
            )
        )
        self.processor = AutoProcessor.from_pretrained(llm_model_path, trust_remote_code=True)
        tokenizer_module = load_tokenizer(self.model_args)
        self.tokenizer = tokenizer_module["tokenizer"]
        self.template = get_template_and_fix_tokenizer(self.tokenizer, self.data_args)
        self.template.mm_plugin.expand_mm_tokens = False

        engine_args = {
            "model": self.model_args.model_name_or_path,
            "trust_remote_code": True,
            "dtype": self.model_args.infer_dtype,
            "max_model_len": 16384,
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.9,
            "rope_scaling": {"type": "dynamic", "factor": 4.0},
            "disable_log_stats": True,
            "limit_mm_per_prompt": {"image": 4}
        }
        self.llm_engine = LLM(**engine_args)

        self.sampling_params = SamplingParams(
            temperature=0.1,
            max_tokens=4096,
            skip_special_tokens=False, # 必须为 False 才能生成您的特殊 token
            seed=42,
            stop_token_ids=[self.processor.tokenizer.eos_token_id] # 使用 processor 获取停止 token
        )

        print("1/3: LLM Engine loaded.")

    def _init_codec_models(self, codec_config_path: str, rt_config_path: str):
        """加载并初始化 Edge Codec 和 RT Codec 模型。"""
        print(f"2/3: Loading Edge Codec using config: {codec_config_path}")
        codec_config = read_yaml_config(codec_config_path)
        self.num_q = codec_config["codec"]["num_quantizers"] # 动态获取 num_q
        
        if codec_config["codec"]["is_transformer"]:
            self.codec_model = VQTransformer(**codec_config['codec']).to(self.device)
        else:
            self.codec_model = VQAutoCodec(
                codec_config["codec"]["input_dim"], codec_config["codec"]["latent_dim"],
                codec_config["codec"]["hidden_dim"], codec_config["sample"]["sample_point_num"],
                codec_config["codec"]["num_layers"], codec_config["codec"]["n_heads"],
                codec_config["codebook"]["size"], codec_config["codec"]["num_quantizers"],
                codec_config["codebook"]["decay"], codec_config["codebook"]["is_fsq"],
                [8] * codec_config["codec"]["latent_dim"],
            ).to(self.device)
        
        checkpoint_path = codec_config["test"]["checkpoint_path"]
        print(f"   - Loading weights from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        # 兼容不同checkpoint格式
        if "model_state_dict" in checkpoint:
            self.codec_model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.codec_model.load_state_dict(checkpoint)
        self.codec_model.eval()
        print("2/3: Edge Codec loaded.")

        print(f"3/3: Loading RT Codec using config: {rt_config_path}")
        rt_config = read_yaml_config(rt_config_path)
        self.rt_model = VQRTCodec(rt_config).to(self.device)
        
        rt_checkpoint_path = rt_config["test"]["test_model_path"]
        print(f"   - Loading weights from: {rt_checkpoint_path}")
        rt_checkpoint = torch.load(rt_checkpoint_path, map_location=self.device)
        self.rt_model.load_state_dict(rt_checkpoint)
        self.rt_model.eval()
        print("3/3: RT Codec loaded.")

    def predict(self, image_path: str) -> Optional[Dict]:
        """
        对单张图片执行完整的端到端推理。
        """
        # === 阶段 1: LLM 推理 (Image -> Text Tokens) ===
        print(f"\n--- Running Stage 1: LLM Inference for {os.path.basename(image_path)} ---")
        generated_text = self._run_llm_inference(image_path)
        if not generated_text:
            return None
        
        # === 阶段 2: 文本解析 (Text Tokens -> Parsed JSON with indices) ===
        print("\n--- Running Stage 2: Parsing generated text ---")
        parsed_data = parse_garment_spidx_pure([generated_text], num_q=self.num_q)[0]
        if not parsed_data or not parsed_data.get("panels"):
            print("Error: Failed to parse valid panels from LLM output.")
            return {"error": "Parsing failed", "raw_output": generated_text}
        
        # === 阶段 3: 几何解码 (Indices -> Floating Point Geometry) ===
        print("\n--- Running Stage 3: Converting indices to geometry ---")
        final_gcd_json = self._convert_indices_to_gcd(parsed_data)
        
        # 添加一些元数据
        final_gcd_json["source_image"] = os.path.abspath(image_path)
        final_gcd_json["raw_llm_output"] = generated_text
        
        return final_gcd_json

    # def _run_llm_inference(self, image_path: str) -> Optional[str]:
    #     try:
    #         image = Image.open(image_path).convert("RGB")
    #     except Exception as e:
    #         print(f"Error opening image {image_path}: {e}")
    #         return None

    #     query = self.template.mm_plugin.get_raw_image_token()
    #     prompt_ids, _ = self.template.encode_multimodal_example(
    #         source=[{"from": "user", "value": query}],
    #     )
    #     multi_modal_data = {
    #         "image": self.template.mm_plugin._regularize_images([image], image_max_pixels=112896)
    #     }
    #     inputs = {"prompt_token_ids": prompt_ids, "multi_modal_data": multi_modal_data}

    #     sampling_params = SamplingParams(
    #         temperature=self.generating_args.temperature,
    #         max_tokens=self.generating_args.max_new_tokens,
    #         stop_token_ids=self.template.get_stop_token_ids(self.tokenizer),
    #         skip_special_tokens=False,
    #         seed=42,
    #     )

    #     results = self.llm_engine.generate([inputs], sampling_params)
    #     return results[0].outputs[0].text


    # def _run_llm_inference(self, image_path: str) -> Optional[str]:
    #     try:
    #         image = Image.open(image_path).convert("RGB")
    #     except Exception as e:
    #         print(f"Error opening image {image_path}: {e}")
    #         return None
    #     query = "<image>" 
    #     prompt_ids, _ = self.template.encode_multimodal_example(
    #         source=[{"from": "user", "value": query}],
    #         image=[image] 
    #     )
    #     inputs = {"prompt_token_ids": prompt_ids}
    #     multi_modal_data = {
    #         "image": self.template.mm_plugin._regularize_images([image], image_max_pixels=112896)
    #     }
    #     inputs["multi_modal_data"] = multi_modal_data
    #     sampling_params = SamplingParams(
    #         temperature=self.generating_args.temperature,
    #         max_tokens=self.generating_args.max_new_tokens,
    #         stop_token_ids=self.template.get_stop_token_ids(self.tokenizer),
    #         skip_special_tokens=False,
    #         seed=42,
    #     )
    #     results = self.llm_engine.generate([inputs], sampling_params)
    #     return results[0].outputs[0].text

    def _run_llm_inference(self, image_path: str) -> Optional[str]:
        """使用 AutoProcessor 和 vLLM 执行推理 (已重构)。"""
        try:
            image = Image.open(image_path).convert("RGB").copy()
        except Exception as e:
            print(f"Error opening image {image_path}: {e}")
            return None
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "You are a professional fashion designer specializing in technical pattern drafting. When provided with garment images or descriptions, generate corresponding sewing patterns."},
                    {"type": "image"},
                ]
            }
        ]
        final_prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = {
            "prompt": final_prompt,
            "multi_modal_data": {"image": [image]}
        }
        results = self.llm_engine.generate([inputs], self.sampling_params)
        return results[0].outputs[0].text

    @torch.no_grad()
    def _convert_indices_to_gcd(self, parsed_data: Dict) -> Dict:
        """核心解码逻辑，移植自 convert.py。"""
        gcd_result = {
            "pattern": {"panels": {}, "stitches": [], "panel_order": []},
            "parameters": {}, "parameter_order": [],
            "properties": {
                "curvature_coords": "relative", "normalize_panel_translation": False,
                "normalized_edge_loops": True, "units_in_meter": 100,
            },
        }
        
        pred_scale = parsed_data.get("scale", 1.0)
        if pred_scale == -1: pred_scale = 1.0 # 如果模型没生成scale，使用默认值

        for panel_data in parsed_data["panels"]:
            panel_name = panel_data["name"]
            edge_indices = panel_data["edges"]
            location_indices = panel_data.get("location", [])

            if not edge_indices:
                print(f"警告: 面板 '{panel_name}' 没有有效的边线索引，将跳过其几何解码。")
                panel_translation = [0.0, 0.0, 0.0]
                panel_rotation = [1.0, 0.0, 0.0, 0.0]
                if self.rt_model and location_indices:
                    rt_indices_tensor = torch.tensor([location_indices], device=self.device)
                    rt_output = self.rt_model.decoder(self.rt_model.codebook.quantizer.get_output_from_indices(rt_indices_tensor))
                    rt_values = rt_output[0].cpu().numpy().tolist()
                    panel_translation = rt_values[:3]
                    panel_rotation = rt_values[3:]
                gcd_panel = {"translation": panel_translation, "rotation": panel_rotation, "vertices": [], "edges": []}
                gcd_result["pattern"]["panels"][panel_name] = gcd_panel
                gcd_result["pattern"]["panel_order"].append(panel_name)
                continue 

            indices_tensor = torch.tensor(edge_indices, device=self.device)
            if self.codec_model.codebook.is_fsq:
                quantized = self.codec_model.codebook.residual_vq.indices_to_codes(indices_tensor)
            else:
                quantized = self.codec_model.codebook.residual_vq.get_output_from_indices(indices_tensor)
            
            decoded_edges = self.codec_model.decoder(quantized)
            
            for key in decoded_edges:
                if torch.is_tensor(decoded_edges[key]):
                    decoded_edges[key] = decoded_edges[key] * pred_scale

            # 重建顶点
            start_points = decoded_edges["start_point"].cpu().numpy()
            end_points = decoded_edges["end_point"].cpu().numpy()
            vertices = []
            edge_count = len(edge_indices)
            for i in range(edge_count):
                avg_vertex = (start_points[i] + end_points[(i - 1) % edge_count]) / 2
                vertices.append(avg_vertex.tolist())
            
            # --- 解码位姿 (RT) ---
            panel_translation = [0.0, 0.0, 0.0]
            panel_rotation = [1.0, 0.0, 0.0, 0.0] # 默认旋转 (w,x,y,z)
            if self.rt_model and location_indices:
                rt_indices_tensor = torch.tensor([location_indices], device=self.device)
                rt_output = self.rt_model.decoder(self.rt_model.codebook.quantizer.get_output_from_indices(rt_indices_tensor))
                rt_values = rt_output[0].cpu().numpy().tolist()
                panel_translation = rt_values[:3]
                panel_rotation = rt_values[3:]

            # 构建 GCD panel
            gcd_panel = {"translation": panel_translation, "rotation": panel_rotation, "vertices": vertices, "edges": []}

            for i in range(edge_count):
                gcd_edge = {"endpoints": [i, (i + 1) % edge_count]}
                line_type_idx = torch.argmax(decoded_edges["line_type_logits"][i], dim=0).item()

                if line_type_idx == 1: # Bezier -> cubic
                    start_point = np.array(vertices[i])
                    end_point = np.array(vertices[(i + 1) % edge_count])
                    control_point_1 = decoded_edges["cubic_control_point1"][i].cpu().numpy()
                    control_point_2 = decoded_edges["cubic_control_point2"][i].cpu().numpy()
                    
                    vec = end_point - start_point
                    vec_length = np.linalg.norm(vec)
                    if vec_length > 0:
                        angle = np.arctan2(vec[1], vec[0])
                        rotation_matrix = np.array([[np.cos(-angle), -np.sin(-angle)], [np.sin(-angle), np.cos(-angle)]])
                        
                        normalized_control_1 = (rotation_matrix @ (control_point_1 - start_point)) / vec_length
                        normalized_control_2 = (rotation_matrix @ (control_point_2 - start_point)) / vec_length
                        
                        gcd_edge["curvature"] = {
                            "type": "cubic",
                            "params": [normalized_control_1.tolist(), normalized_control_2.tolist()]
                        }
                elif line_type_idx == 2: # Arc -> circle
                    gcd_edge["curvature"] = {
                        "type": "circle",
                        "params": [
                            decoded_edges["arc_radius"][i].cpu().item(),
                            int(decoded_edges["arc_large_flag"][i].cpu().item() > 0.5),
                            int(decoded_edges["arc_sweep_flag"][i].cpu().item() > 0.5)
                        ]
                    }
                # line_type 0 (直线) 不需要 "curvature" 字段
                
                if "curvature" in gcd_edge:
                    gcd_panel["edges"].append(gcd_edge)
                else: 
                    gcd_panel["edges"].append({"endpoints": [i, (i + 1) % edge_count]})


            gcd_result["pattern"]["panels"][panel_name] = gcd_panel
            gcd_result["pattern"]["panel_order"].append(panel_name)

        for stitch in parsed_data.get("stitches", []):
            gcd_result['pattern']['stitches'].append([
                {"panel": stitch['panel1'], "edge": stitch['edge1']},
                {"panel": stitch['panel2'], "edge": stitch['edge2']}
            ])
            
        return gcd_result


def main(
    llm_model_path: str,
    codec_config_path: str,
    rt_config_path: str,
    image_path: str,
    output_path: str,
    device: str = "cuda:0",
):
    """
    对单张服装图片进行端到端的几何结构预测。

    Args:
        llm_model_path (str): LLaVA 微调模型的路径。
        codec_config_path (str): 边线编解码器 (Edge Codec) 的 YAML 配置文件路径。
        rt_config_path (str): 位姿编解码器 (RT Codec) 的 YAML 配置文件路径。
        image_path (str): 输入的单张图片路径。
        output_path (str): 输出的最终 GCD 格式 JSON 文件路径。
        device (str, optional): 用于 Codec 模型的设备。默认为 "cuda:0"。
    """
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    predictor = GarmentPredictor(
        llm_model_path=llm_model_path,
        codec_config_path=codec_config_path,
        rt_config_path=rt_config_path,
        device=device,
    )
    
    # 执行预测
    result_json = predictor.predict(image_path=image_path)
    
    # 保存结果
    if result_json:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result_json, f, indent=4, ensure_ascii=False)
        print(f"\n✅ Prediction successful! Result saved to: {output_path}")
    else:
        print(f"\n❌ Prediction failed for image {image_path}. No output file was created.")


if __name__ == "__main__":
    # 使用 Fire 创建强大的命令行接口
    fire.Fire(main)
