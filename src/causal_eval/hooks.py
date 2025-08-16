"""Hook classes for causal interventions and ablations in language model activations."""

import logging
import numpy as np
import torch

from transformers import (
    PreTrainedModel,
)

from src.transport_operator import TransportOperator

logger = logging.getLogger(__name__)


class ZeroHook:
    """Hook that zeros out activations at specific positions in a layer."""

    def __init__(self, layer_name: str, target_j_positions: list[int]):
        self.layer_name = layer_name
        self.target_j_positions = target_j_positions
        self.hook_handle = None

    def apply(self, model: PreTrainedModel):
        """Apply hook that zeros out activations at target positions."""
        # Get the target module
        target_module = model
        for attr in self.layer_name.split("."):
            target_module = getattr(target_module, attr)

        def zero_hook(module, input_tensors, output):
            """Hook function that zeros activations at specific positions."""
            try:
                if isinstance(output, torch.Tensor):
                    # Clone the output to avoid modifying in-place
                    modified_output = output.clone()
                    # Zero out the specified positions
                    for j_position in self.target_j_positions:
                        if j_position < modified_output.size(1):
                            modified_output[:, j_position, :] = 0.0
                    return modified_output
                elif isinstance(output, tuple):
                    # For layers that return tuples (like attention), modify the first element
                    modified_output = list(output)
                    modified_output[0] = output[0].clone()
                    # Zero out the specified positions
                    for j_position in self.target_j_positions:
                        if j_position < modified_output[0].size(1):
                            modified_output[0][:, j_position, :] = 0.0
                    return tuple(modified_output)
                else:
                    logger.warning(f"Unsupported output type: {type(output)}")
                    return output
            except Exception as e:
                logger.exception(f"Zero operation failed: {e}.")
                raise RuntimeError("Zero operation failed")

        self.hook_handle = target_module.register_forward_hook(zero_hook)
        logger.info(
            f"Applied zero hook to layer '{self.layer_name}' at positions {self.target_j_positions}"
        )

    def remove(self):
        """Remove the hook."""
        if self.hook_handle:
            self.hook_handle.remove()
            self.hook_handle = None


class TransportHook:
    """Hook for transport operator interventions that captures from source and modifies target layer."""

    def __init__(
        self,
        name: str,
        source_layer: str,
        target_layer: str,
        transport_operator: TransportOperator,
        target_j_positions: list[int],
    ):
        self.name = name
        self.source_layer = source_layer
        self.target_layer = target_layer
        self.transport_operator = transport_operator
        self.captured_activation = None
        self.source_hook_handle = None
        self.target_hook_handle = None
        self.target_j_positions = target_j_positions

    def apply(self, model: PreTrainedModel):
        """Apply both source capture and target modification hooks."""
        if self.source_layer is not None:
            # Hook for capturing from source layer
            source_module = model
            for attr in self.source_layer.split("."):
                source_module = getattr(source_module, attr)

            def capture_hook(module, input_tensors, output):
                # Store the hidden states for transport operator
                if isinstance(output, torch.Tensor):
                    self.captured_activation = output[
                        :, self.target_j_positions, :
                    ].detach()
                elif isinstance(output, tuple):
                    # For layers that return tuples (like attention), take the first element
                    self.captured_activation = output[0][
                        :, self.target_j_positions, :
                    ].detach()
                return output

            self.source_hook_handle = source_module.register_forward_hook(capture_hook)

        # Hook for modifying target layer
        target_module = model
        for attr in self.target_layer.split("."):
            target_module = getattr(target_module, attr)

        def transport_hook(module, input_tensors, output):
            if self.source_layer is None:
                # Use null vector as input to transport operator
                if isinstance(output, torch.Tensor):
                    batch_size, _, _ = output.shape
                    # Get input features from transport matrix shape
                    input_features = (
                        self.transport_operator.get_transport_matrix().shape[0]
                    )
                    null_input = torch.zeros(
                        batch_size,
                        len(self.target_j_positions),
                        input_features,
                        dtype=output.dtype,
                        device=output.device,
                    )
                else:
                    batch_size, seq_len, hidden_size = output[0].shape

                    input_features = (
                        self.transport_operator.get_transport_matrix().shape[0]
                    )
                    null_input = torch.zeros(
                        batch_size,
                        len(self.target_j_positions),
                        input_features,
                        dtype=output[0].dtype,
                        device=output[0].device,
                    )

                transport_input = null_input
            else:
                # Use captured activation from source layer
                if self.captured_activation is None:
                    raise RuntimeError("No captured activation available for transport")
                transport_input = self.captured_activation

            try:
                # Apply transport operator
                original_shape = transport_input.shape
                # Reshape to (batch_size * seq_len, hidden_size) for transport operator
                flat_input = transport_input.view(-1, transport_input.shape[-1])

                # Convert to numpy for transport operator
                flat_input_np = flat_input.float().cpu().numpy()

                # Apply transport operator
                transported_np = self.transport_operator.predict(flat_input_np)

                # Convert back to torch and reshape
                transported = torch.from_numpy(transported_np.astype(np.float32))
                transported = transported.to(transport_input.device)
                transported = transported.to(transport_input.dtype)
                transported = transported.view(
                    original_shape[0], len(self.target_j_positions), -1
                )

                # Replace the output
                if isinstance(output, torch.Tensor):
                    for j, j_position in enumerate(self.target_j_positions):
                        output[:, j_position, :] = transported[:, j, :]
                    return output
                elif isinstance(output, tuple):
                    modified_output = list(output)
                    modified_output[0] = output[0].clone()
                    for j, j_position in enumerate(self.target_j_positions):
                        modified_output[0][:, j_position, :] = transported[:, j, :]
                    return tuple(modified_output)
                else:
                    raise RuntimeError("Unsupported output type")

            except Exception as e:
                logger.exception(f"Transport operation failed: {e}.")
                raise RuntimeError("Transport operation failed")

        self.target_hook_handle = target_module.register_forward_hook(transport_hook)
        logger.info(
            f"Applied transport hook '{self.name}': {self.source_layer} -> {self.target_layer}"
        )

    def remove(self):
        """Remove both hooks."""
        if self.source_hook_handle:
            self.source_hook_handle.remove()
            self.source_hook_handle = None
        if self.target_hook_handle:
            self.target_hook_handle.remove()
            self.target_hook_handle = None
        self.captured_activation = None


def create_j_hook_family(
    transport_operator,
    source_layer: str,
    target_layer: str,
    js: list[list[int]],
    prefix: str,
) -> dict[str, TransportHook]:
    """Create a family of transport operator hooks for a specific layer."""
    hooks = {}
    for j in js:
        hooks[f"{prefix}_{str(j)}"] = create_transport_hook(
            transport_operator,
            source_layer=source_layer,
            target_layer=target_layer,
            j_positions=j,
        )
    return hooks


def create_transport_hook(
    transport_operator, source_layer: str, target_layer: str, j_positions: list[int]
) -> TransportHook:
    """Create a transport operator hook that captures from source and transports to target."""
    return TransportHook(
        "transport_intervention",
        source_layer,
        target_layer,
        transport_operator,
        j_positions,
    )


def create_zero_hook(layer_name: str, j_positions: list[int]) -> ZeroHook:
    """Create a zero hook that zeros out activations at specific positions."""
    return ZeroHook(layer_name, j_positions)


def create_zero_hook_family(
    layer_name: str,
    js: list[list[int]],
    prefix: str,
) -> dict[str, ZeroHook]:
    """Create a family of zero hooks for a specific layer."""
    hooks = {}
    for j in js:
        hooks[f"{prefix}_{str(j)}"] = create_zero_hook(layer_name, j)
    return hooks
