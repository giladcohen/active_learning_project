import logging
import numpy as np
import torch
from typing import List, Union
from art.estimators.classification.pytorch import PyTorchClassifier

logger = logging.getLogger(__name__)


class PyTorchClassifierSpecific(PyTorchClassifier):  # lgtm [py/missing-call-to-init]
    def __init__(self, *args, **kwargs):
        self.fields = kwargs.pop('fields')
        super().__init__(*args, **kwargs)

    def predict_specific(  # pylint: disable=W0221
            self, x: np.ndarray, field: str, batch_size: int = 128, training_mode: bool = False, **kwargs
    ) -> np.ndarray:

        assert field in self.fields
        ind = self.fields.index(field)

        # Set model mode
        self._model.train(mode=training_mode)

        # Apply preprocessing
        x_preprocessed, _ = self._apply_preprocessing(x, y=None, fit=False)

        results_list = []

        # Run prediction with batch processing
        num_batch = int(np.ceil(len(x_preprocessed) / float(batch_size)))
        for m in range(num_batch):
            # Batch indexes
            begin, end = (
                m * batch_size,
                min((m + 1) * batch_size, x_preprocessed.shape[0]),
            )

            with torch.no_grad():
                model_outputs = self._model(torch.from_numpy(x_preprocessed[begin:end]).to(self._device))
            output = model_outputs[ind]
            output = output.detach().cpu().numpy().astype(np.float32)
            if len(output.shape) == 1:
                output = np.expand_dims(output.detach().cpu().numpy(), axis=1).astype(np.float32)

            results_list.append(output)

        results = np.vstack(results_list)

        # Apply postprocessing
        predictions = self._apply_postprocessing(preds=results, fit=False)

        return predictions

    def class_gradient_specific(  # pylint: disable=W0221
            self, x: np.ndarray, label: Union[int, List[int], None] = None, field: str = 'logits', training_mode: bool = False, **kwargs
    ) -> np.ndarray:

        assert field in self.fields
        assert field == 'logits'
        ind = self.fields.index(field)
        self._model.train(mode=training_mode)

        # Backpropagation through RNN modules in eval mode raises RuntimeError due to cudnn issues and require training
        # mode, i.e. RuntimeError: cudnn RNN backward can only be called in training mode. Therefore, if the model is
        # an RNN type we always use training mode but freeze batch-norm and dropout layers if training_mode=False.
        if self.is_rnn:
            self._model.train(mode=True)
            if not training_mode:
                logger.debug(
                    "Freezing batch-norm and dropout layers for gradient calculation in train mode with eval parameters"
                    "of batch-norm and dropout."
                )
                self.set_batchnorm(train=False)
                self.set_dropout(train=False)

        if not (
                (label is None)
                or (isinstance(label, (int, np.integer)) and label in range(self._nb_classes))
                or (
                        isinstance(label, np.ndarray)
                        and len(label.shape) == 1
                        and (label < self._nb_classes).all()
                        and label.shape[0] == x.shape[0]
                )
        ):
            raise ValueError("Label %s is out of range." % label)

        # Apply preprocessing
        if self.all_framework_preprocessing:
            x_grad = torch.from_numpy(x).to(self._device)
            if self._layer_idx_gradients < 0:
                x_grad.requires_grad = True
            x_input, _ = self._apply_preprocessing(x_grad, y=None, fit=False, no_grad=False)
        else:
            x_preprocessed, _ = self._apply_preprocessing(x, y=None, fit=False, no_grad=True)
            x_grad = torch.from_numpy(x_preprocessed).to(self._device)
            if self._layer_idx_gradients < 0:
                x_grad.requires_grad = True
            x_input = x_grad

        # Run prediction
        model_outputs = self._model(x_input)

        # Set where to get gradient
        if self._layer_idx_gradients >= 0:
            input_grad = model_outputs[self._layer_idx_gradients]
        else:
            input_grad = x_grad

        # Set where to get gradient from
        preds = model_outputs[ind]

        # Compute the gradient
        grads = []

        def save_grad():
            def hook(grad):
                grads.append(grad.cpu().numpy().copy())
                grad.data.zero_()

            return hook

        input_grad.register_hook(save_grad())

        self._model.zero_grad()
        if label is None:
            if len(preds.shape) == 1 or preds.shape[1] == 1:
                num_outputs = 1
            else:
                num_outputs = self.nb_classes

            for i in range(num_outputs):
                torch.autograd.backward(
                    preds[:, i],
                    torch.tensor([1.0] * len(preds[:, 0])).to(self._device),
                    retain_graph=True,
                )

        elif isinstance(label, (int, np.integer)):
            torch.autograd.backward(
                preds[:, label],
                torch.tensor([1.0] * len(preds[:, 0])).to(self._device),
                retain_graph=True,
            )
        else:
            unique_label = list(np.unique(label))
            for i in unique_label:
                torch.autograd.backward(
                    preds[:, i],
                    torch.tensor([1.0] * len(preds[:, 0])).to(self._device),
                    retain_graph=True,
                )

            grads = np.swapaxes(np.array(grads), 0, 1)
            lst = [unique_label.index(i) for i in label]
            grads = grads[np.arange(len(grads)), lst]

            grads = grads[None, ...]

        grads = np.swapaxes(np.array(grads), 0, 1)
        if not self.all_framework_preprocessing:
            grads = self._apply_preprocessing_gradient(x, grads)

        return grads

    def _make_model_wrapper(self, model: "torch.nn.Module") -> "torch.nn.Module":
        # Try to import PyTorch and create an internal class that acts like a model wrapper extending torch.nn.Module
        try:
            import torch.nn as nn

            # Define model wrapping class only if not defined before
            if not hasattr(self, "_model_wrapper"):

                class ModelWrapper(nn.Module):
                    """
                    This is a wrapper for the input model.
                    """

                    import torch  # lgtm [py/repeated-import]

                    def __init__(self, model: torch.nn.Module, fields: List[str]):
                        """
                        Initialization by storing the input model.

                        :param model: PyTorch model. The forward function of the model must return the logit output.
                        """
                        super().__init__()
                        self._model = model
                        self.fields = fields

                    # pylint: disable=W0221
                    # disable pylint because of API requirements for function
                    def forward(self, x):
                        """
                        This is where we get outputs from the input model.

                        :param x: Input data.
                        :type x: `torch.Tensor`
                        :return: a list of output layers, where the last 2 layers are logit and final outputs.
                        :rtype: `list`
                        """
                        # pylint: disable=W0212
                        # disable pylint because access to _model required
                        import torch.nn as nn

                        result = []
                        if isinstance(self._model, nn.Sequential):
                            for _, module_ in self._model._modules.items():
                                x = module_(x)
                                result.append(x)

                        elif isinstance(self._model, nn.Module):
                            x = self._model(x)
                            for field in self.fields:
                                result.append(x[field])

                        else:
                            raise TypeError("The input model must inherit from `nn.Module`.")

                        return result

                    @property
                    def get_layers(self) -> List[str]:
                        """
                        Return the hidden layers in the model, if applicable.

                        :return: The hidden layers in the model, input and output layers excluded.

                        .. warning:: `get_layers` tries to infer the internal structure of the model.
                                     This feature comes with no guarantees on the correctness of the result.
                                     The intended order of the layers tries to match their order in the model, but this
                                     is not guaranteed either. In addition, the function can only infer the internal
                                     layers if the input model is of type `nn.Sequential`, otherwise, it will only
                                     return the logit layer.
                        """
                        import torch.nn as nn

                        result = []
                        if isinstance(self._model, nn.Sequential):
                            # pylint: disable=W0212
                            # disable pylint because access to _modules required
                            for name, module_ in self._model._modules.items():  # type: ignore
                                result.append(name + "_" + str(module_))

                        elif isinstance(self._model, nn.Module):
                            result.append("final_layer")

                        else:
                            raise TypeError("The input model must inherit from `nn.Module`.")
                        logger.info(
                            "Inferred %i hidden layers on PyTorch classifier.",
                            len(result),
                        )

                        return result

                # Set newly created class as private attribute
                self._model_wrapper = ModelWrapper

            # Use model wrapping class to wrap the PyTorch model received as argument
            return self._model_wrapper(model, self.fields)

        except ImportError:
            raise ImportError("Could not find PyTorch (`torch`) installation.") from ImportError

    def __repr__(self):
        repr_ = (
                "%s(model=%r, loss=%r, optimizer=%r, input_shape=%r, nb_classes=%r, channels_first=%r, "
                "clip_values=%r, preprocessing_defences=%r, postprocessing_defences=%r, preprocessing=%r, fields=%r)"
                % (
                    self.__module__ + "." + self.__class__.__name__,
                    self._model,
                    self._loss,
                    self._optimizer,
                    self._input_shape,
                    self.nb_classes,
                    self.channels_first,
                    self.clip_values,
                    self.preprocessing_defences,
                    self.postprocessing_defences,
                    self.preprocessing,
                    self.fields
                )
        )

        return repr_
