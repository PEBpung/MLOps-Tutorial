import numpy as np
from numpy.typing import NDArray
import typing as t

import bentoml
from bentoml.io import NumpyNdarray

mnist_runner = bentoml.pytorch.get("pytorch_mnist").to_runner()

svc = bentoml.Service(name="pytorch_mnist_demo", runners=[mnist_runner])

@svc.api(
    input=NumpyNdarray(dtype="float32", enforce_dtype=True),
    output=NumpyNdarray(dtype="int64"),
)
async def predict_ndarray(inp: NDArray[t.Any]) -> NDArray[t.Any]:
	# shape 체크
    assert inp.shape == (28, 28)
    # batch와 channel dimension 추가
    inp = np.expand_dims(inp, (0, 1))
    output_tensor = await mnist_runner.async_run(inp)
    return output_tensor.detach().cpu().numpy()