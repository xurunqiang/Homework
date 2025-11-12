from pathlib import Path
from typing import List, Optional, Dict
import warnings


# ImageNet类别到中文智能分类的映射
CATEGORY_MAPPING: Dict[str, str] = {
    # 动物类
    "dog": "动物",
    "cat": "动物",
    "bird": "动物",
    "horse": "动物",
    "cow": "动物",
    "elephant": "动物",
    "bear": "动物",
    "zebra": "动物",
    "giraffe": "动物",
    "sheep": "动物",
    "goat": "动物",
    "pig": "动物",
    "chicken": "动物",
    "rooster": "动物",
    "hen": "动物",
    "turkey": "动物",
    "duck": "动物",
    "goose": "动物",
    "rabbit": "动物",
    "hamster": "动物",
    "mouse": "动物",
    "rat": "动物",
    "squirrel": "动物",
    "fox": "动物",
    "wolf": "动物",
    "tiger": "动物",
    "lion": "动物",
    "leopard": "动物",
    "jaguar": "动物",
    "panther": "动物",
    "monkey": "动物",
    "ape": "动物",
    "gorilla": "动物",
    "chimpanzee": "动物",
    "kangaroo": "动物",
    "koala": "动物",
    "panda": "动物",
    "penguin": "动物",
    "seal": "动物",
    "whale": "动物",
    "dolphin": "动物",
    "shark": "动物",
    "fish": "动物",
    "snake": "动物",
    "lizard": "动物",
    "turtle": "动物",
    "frog": "动物",
    "toad": "动物",
    "spider": "动物",
    "insect": "动物",
    "butterfly": "动物",
    "bee": "动物",
    "ant": "动物",
    
    # 植物类
    "tree": "植物",
    "flower": "植物",
    "rose": "植物",
    "daisy": "植物",
    "tulip": "植物",
    "sunflower": "植物",
    "orchid": "植物",
    "lily": "植物",
    "cactus": "植物",
    "palm": "植物",
    "oak": "植物",
    "pine": "植物",
    "bamboo": "植物",
    "grass": "植物",
    "leaf": "植物",
    "branch": "植物",
    "fruit": "植物",
    "apple": "植物",
    "banana": "植物",
    "orange": "植物",
    "grape": "植物",
    "strawberry": "植物",
    "watermelon": "植物",
    "vegetable": "植物",
    "carrot": "植物",
    "potato": "植物",
    "tomato": "植物",
    "corn": "植物",
    "wheat": "植物",
    "rice": "植物",
    
    # 风景类
    "mountain": "风景",
    "mount": "风景",
    "alp": "风景",
    "alpine": "风景",
    "hill": "风景",
    "valley": "风景",
    "vale": "风景",
    "lake": "风景",
    "lakeside": "风景",
    "lakeshore": "风景",
    "river": "风景",
    "stream": "风景",
    "brook": "风景",
    "creek": "风景",
    "ocean": "风景",
    "sea": "风景",
    "seashore": "风景",
    "seacoast": "风景",
    "beach": "风景",
    "coast": "风景",
    "coastline": "风景",
    "shore": "风景",
    "shoreline": "风景",
    "island": "风景",
    "isle": "风景",
    "forest": "风景",
    "woodland": "风景",
    "woods": "风景",
    "jungle": "风景",
    "desert": "风景",
    "snow": "风景",
    "snowfield": "风景",
    "ice": "风景",
    "glacier": "风景",
    "sky": "风景",
    "cloud": "风景",
    "sunset": "风景",
    "sunrise": "风景",
    "sun": "风景",
    "moon": "风景",
    "star": "风景",
    "rainbow": "风景",
    "waterfall": "风景",
    "cascade": "风景",
    "canyon": "风景",
    "gorge": "风景",
    "cliff": "风景",
    "bluff": "风景",
    "cave": "风景",
    "cavern": "风景",
    "volcano": "风景",
    "meadow": "风景",
    "prairie": "风景",
    "grassland": "风景",
    "tundra": "风景",
    "marsh": "风景",
    "swamp": "风景",
    "wetland": "风景",
    "pond": "风景",
    "lagoon": "风景",
    "bay": "风景",
    "gulf": "风景",
    "fjord": "风景",
    "plateau": "风景",
    "plain": "风景",
    "field": "风景",
    "countryside": "风景",
    "rural": "风景",
    "landscape": "风景",
    "scenery": "风景",
    "vista": "风景",
    "view": "风景",
    "horizon": "风景",
    
    # 建筑类
    "building": "建筑",
    "house": "建筑",
    "cottage": "建筑",
    "mansion": "建筑",
    "palace": "建筑",
    "castle": "建筑",
    "tower": "建筑",
    "bridge": "建筑",
    "church": "建筑",
    "temple": "建筑",
    "mosque": "建筑",
    "cathedral": "建筑",
    "skyscraper": "建筑",
    "office": "建筑",
    "school": "建筑",
    "hospital": "建筑",
    "hotel": "建筑",
    "restaurant": "建筑",
    "shop": "建筑",
    "store": "建筑",
    "market": "建筑",
    "street": "建筑",
    "road": "建筑",
    "path": "建筑",
    "park": "建筑",
    "garden": "建筑",
    
    # 生活类
    "person": "生活",
    "people": "生活",
    "man": "生活",
    "woman": "生活",
    "child": "生活",
    "baby": "生活",
    "family": "生活",
    "food": "生活",
    "meal": "生活",
    "breakfast": "生活",
    "lunch": "生活",
    "dinner": "生活",
    "cake": "生活",
    "bread": "生活",
    "pizza": "生活",
    "burger": "生活",
    "coffee": "生活",
    "tea": "生活",
    "drink": "生活",
    "cup": "生活",
    "plate": "生活",
    "bowl": "生活",
    "bottle": "生活",
    "glass": "生活",
    "furniture": "生活",
    "chair": "生活",
    "table": "生活",
    "bed": "生活",
    "sofa": "生活",
    "couch": "生活",
    "desk": "生活",
    "lamp": "生活",
    "light": "生活",
    "clock": "生活",
    "watch": "生活",
    "phone": "生活",
    "computer": "生活",
    "laptop": "生活",
    "keyboard": "生活",
    "mouse": "生活",
    "screen": "生活",
    "television": "生活",
    "tv": "生活",
    "camera": "生活",
    "book": "生活",
    "newspaper": "生活",
    "magazine": "生活",
    "bag": "生活",
    "backpack": "生活",
    "suitcase": "生活",
    "wallet": "生活",
    "purse": "生活",
    "clothing": "生活",
    "shirt": "生活",
    "dress": "生活",
    "jacket": "生活",
    "coat": "生活",
    "hat": "生活",
    "shoe": "生活",
    "boot": "生活",
    "sneaker": "生活",
    "vehicle": "生活",
    "car": "生活",
    "truck": "生活",
    "bus": "生活",
    "train": "生活",
    "bicycle": "生活",
    "bike": "生活",
    "motorcycle": "生活",
    "airplane": "生活",
    "plane": "生活",
    "helicopter": "生活",
    "boat": "生活",
    "ship": "生活",
    "sports": "生活",
    "ball": "生活",
    "football": "生活",
    "basketball": "生活",
    "tennis": "生活",
    "golf": "生活",
    "soccer": "生活",
    "baseball": "生活",
    "swimming": "生活",
    "running": "生活",
    "exercise": "生活",
    "game": "生活",
    "toy": "生活",
    "doll": "生活",
    "teddy": "生活",
    "puzzle": "生活",
    "music": "生活",
    "instrument": "生活",
    "guitar": "生活",
    "piano": "生活",
    "violin": "生活",
    "drum": "生活",
    "art": "生活",
    "painting": "生活",
    "drawing": "生活",
    "sculpture": "生活",
    "photo": "生活",
    "picture": "生活",
    "image": "生活",
}


def _map_to_chinese_category(english_label: str) -> str:
    """将英文类别映射到中文智能分类"""
    label_lower = english_label.lower().strip()
    
    # 直接匹配
    if label_lower in CATEGORY_MAPPING:
        return CATEGORY_MAPPING[label_lower]
    
    # 部分匹配（检查是否包含关键词）- 优先匹配较长的关键词
    # 按长度降序排序，优先匹配更具体的词
    sorted_keys = sorted(CATEGORY_MAPPING.keys(), key=len, reverse=True)
    for key in sorted_keys:
        if key in label_lower:
            return CATEGORY_MAPPING[key]
    
    # 反向匹配（检查label是否包含在key中）
    for key, category in CATEGORY_MAPPING.items():
        if label_lower in key:
            return category
    
    # 默认返回"其他"
    return "其他"


def classify_image(image_path: Path) -> Optional[List[str]]:
    """Image classification using torchvision if available.

    Returns top-3 Chinese category labels (动物、植物、风景、生活等).
    If torchvision/torch not installed or weights missing, returns None.
    """
    try:
        import torch
        import torchvision.transforms as T
        from torchvision.models import resnet18, ResNet18_Weights
        from PIL import Image

        # 过滤PyTorch的pin_memory警告（当没有GPU时）
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*pin_memory.*")
            warnings.filterwarnings("ignore", category=UserWarning, module="torch.utils.data")
            
            weights = ResNet18_Weights.DEFAULT
            model = resnet18(weights=weights)
            model.eval()
            
            # 确保模型在CPU上运行（如果没有GPU）
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            
            preprocess = weights.transforms()

            with Image.open(image_path).convert("RGB") as im:
                x = preprocess(im).unsqueeze(0).to(device)
            
            with torch.no_grad():
                y = model(x)
                probs = y.softmax(dim=1)[0]
            
            labels = weights.meta["categories"]
            topk = probs.topk(10)  # 获取top-10以提高准确性，确保能找到风景类别
            
            # 获取英文标签并映射到中文类别
            english_labels = [labels[i] for i in topk.indices.tolist()]
            chinese_categories = []
            seen_categories = set()
            
            # 调试：打印前几个标签（可选，用于调试）
            # print(f"Top labels: {english_labels[:5]}")
            
            for eng_label in english_labels:
                chinese_cat = _map_to_chinese_category(eng_label)
                # 跳过"其他"类别，优先显示具体分类
                if chinese_cat != "其他" and chinese_cat not in seen_categories:
                    chinese_categories.append(chinese_cat)
                    seen_categories.add(chinese_cat)
                    if len(chinese_categories) >= 3:  # 最多返回3个类别
                        break
            
            # 如果没有找到具体分类，返回None（而不是"其他"）
            return chinese_categories if chinese_categories else None
    except Exception:
        return None




