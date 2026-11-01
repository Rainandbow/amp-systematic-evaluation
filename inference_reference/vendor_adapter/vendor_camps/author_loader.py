"""Import pinned inference source without the obsolete training dependencies."""
import importlib.util
import sys
import types
from pathlib import Path
from common import HERE


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def blocked(*args, **kwargs):
    raise RuntimeError("Offline smoke: remote/cache/archive loader is disabled")


def author_modules():
    prefix = "camps_pinned"
    for name in (prefix, prefix + ".model", prefix + ".model.pytorch_pretrained"):
        module = types.ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module
    pp = prefix + ".model.pytorch_pretrained"
    fu = types.ModuleType(pp + ".file_utils")
    fu.WEIGHTS_NAME, fu.CONFIG_NAME = "pytorch_model.bin", "config.json"
    fu.PYTORCH_PRETRAINED_BERT_CACHE = str(HERE / "vendor")
    fu.cached_path = fu.biobert_from_archive = fu.scibert_from_archive = blocked
    sys.modules[fu.__name__] = fu
    # Preserve the author's unfused LayerNorm, independent of optional apex installs.
    sys.modules["apex.normalization.fused_layer_norm"] = None
    modeling = load(pp + ".modeling", HERE / "vendor/modeling.py")
    tokenization = load(pp + ".tokenization", HERE / "vendor/tokenization.py")
    package = sys.modules[pp]
    package.BertModel = modeling.BertModel
    package.BertPreTrainedModel = modeling.BertPreTrainedModel
    head = load(prefix + ".model.model", HERE / "vendor/model.py")
    features = load(prefix + ".features", HERE / "vendor/features.py")
    return modeling, tokenization, head, features
