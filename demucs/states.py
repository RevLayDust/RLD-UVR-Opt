# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Utilities to save and load models.
"""
from contextlib import contextmanager

import functools
import hashlib
import importlib
import inspect
import io
from pathlib import Path
import warnings

from omegaconf import OmegaConf
from diffq import DiffQuantizer, UniformQuantizer, restore_quantized_state
import torch


_SAFE_DEMUCS_GLOBALS = {
    'demucs.demucs.Demucs',
    'demucs.hdemucs.HDemucs',
    'demucs.htdemucs.HTDemucs',
    'demucs.model.Demucs',
    'demucs.model_v2.Demucs',
    'demucs.tasnet.ConvTasNet',
    'demucs.tasnet_v2.ConvTasNet',
}


def get_quantizer(model, args, optimizer=None):
    """Return the quantizer given the XP quantization args."""
    quantizer = None
    if args.diffq:
        quantizer = DiffQuantizer(
            model, min_size=args.min_size, group_size=args.group_size)
        if optimizer is not None:
            quantizer.setup_optimizer(optimizer)
    elif args.qat:
        quantizer = UniformQuantizer(
                model, bits=args.qat, min_size=args.min_size)
    return quantizer


def load_model(path_or_package, strict=False):
    """Load a model from the given serialized model, either given as a dict (already loaded)
    or a path to a file on disk."""
    if isinstance(path_or_package, dict):
        package = path_or_package
    elif isinstance(path_or_package, (str, Path)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            package = _load_checkpoint_package(Path(path_or_package))
    else:
        raise ValueError(f"Invalid type for {path_or_package}.")

    klass = package["klass"]
    args = package["args"]
    kwargs = package["kwargs"]

    if strict:
        model = klass(*args, **kwargs)
    else:
        sig = inspect.signature(klass)
        for key in list(kwargs):
            if key not in sig.parameters:
                warnings.warn("Dropping inexistant parameter " + key)
                del kwargs[key]
        model = klass(*args, **kwargs)

    state = package["state"]

    set_state(model, state)
    return model


def _load_checkpoint_package(path):
    inspector = getattr(torch.serialization, 'get_unsafe_globals_in_checkpoint', None)
    safe_context = getattr(torch.serialization, 'safe_globals', None)
    if inspector is None or safe_context is None:
        raise RuntimeError('Safe legacy Demucs loading requires PyTorch 2.6 or newer.')
    unsafe_globals = set(inspector(path))
    safe_globals, handled_globals = _checkpoint_safe_globals(unsafe_globals)
    unsupported = sorted(unsafe_globals - handled_globals)
    if unsupported:
        raise RuntimeError(
            'Legacy Demucs checkpoint contains unsupported globals: '
            + ', '.join(unsupported)
        )
    with safe_context(safe_globals):
        return torch.load(path, map_location='cpu', weights_only=True)


def _checkpoint_safe_globals(unsafe_globals):
    from fractions import Fraction

    import numpy as np

    safe_globals = []
    handled_globals = set()
    simple_globals = {
        'fractions.Fraction': Fraction,
        'numpy.dtype': np.dtype,
    }
    numpy_core = np._core if hasattr(np, '_core') else np.core
    simple_globals.update({
        'numpy.core.multiarray.scalar': numpy_core.multiarray.scalar,
        'numpy._core.multiarray.scalar': numpy_core.multiarray.scalar,
    })
    for global_name, value in simple_globals.items():
        if global_name in unsafe_globals:
            safe_globals.append((value, global_name))
            handled_globals.add(global_name)

    for global_name in sorted(unsafe_globals & _SAFE_DEMUCS_GLOBALS):
        module_name, attribute = global_name.rsplit('.', 1)
        value = getattr(importlib.import_module(module_name), attribute)
        safe_globals.append((value, global_name))
        handled_globals.add(global_name)

    dtype_classes = {
        type(np.dtype(dtype))
        for dtype in (
            np.bool_, np.int8, np.int16, np.int32, np.int64,
            np.uint8, np.uint16, np.uint32, np.uint64,
            np.float16, np.float32, np.float64,
            np.complex64, np.complex128,
        )
    }
    safe_globals.extend(sorted(dtype_classes, key=lambda value: (value.__module__, value.__qualname__)))
    return safe_globals, handled_globals


def get_state(model, quantizer, half=False):
    """Get the state from a model, potentially with quantization applied.
    If `half` is True, model are stored as half precision, which shouldn't impact performance
    but half the state size."""
    if quantizer is None:
        dtype = torch.half if half else None
        state = {k: p.data.to(device='cpu', dtype=dtype) for k, p in model.state_dict().items()}
    else:
        state = quantizer.get_quantized_state()
        state['__quantized'] = True
    return state


def set_state(model, state, quantizer=None):
    """Set the state on a given model."""
    if state.get('__quantized'):
        if quantizer is not None:
            quantizer.restore_quantized_state(model, state['quantized'])
        else:
            restore_quantized_state(model, state)
    else:
        model.load_state_dict(state)
    return state


def save_with_checksum(content, path):
    """Save the given value on disk, along with a sha256 hash.
    Should be used with the output of either `serialize_model` or `get_state`."""
    buf = io.BytesIO()
    torch.save(content, buf)
    sig = hashlib.sha256(buf.getvalue()).hexdigest()[:8]

    path = path.parent / (path.stem + "-" + sig + path.suffix)
    path.write_bytes(buf.getvalue())


def serialize_model(model, training_args, quantizer=None, half=True):
    args, kwargs = model._init_args_kwargs
    klass = model.__class__

    state = get_state(model, quantizer, half)
    return {
        'klass': klass,
        'args': args,
        'kwargs': kwargs,
        'state': state,
        'training_args': OmegaConf.to_container(training_args, resolve=True),
    }


def copy_state(state):
    return {k: v.cpu().clone() for k, v in state.items()}


@contextmanager
def swap_state(model, state):
    """
    Context manager that swaps the state of a model, e.g:

        # model is in old state
        with swap_state(model, new_state):
            # model in new state
        # model back to old state
    """
    old_state = copy_state(model.state_dict())
    model.load_state_dict(state, strict=False)
    try:
        yield
    finally:
        model.load_state_dict(old_state)


def capture_init(init):
    @functools.wraps(init)
    def __init__(self, *args, **kwargs):
        self._init_args_kwargs = (args, kwargs)
        init(self, *args, **kwargs)

    return __init__
