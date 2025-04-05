.. container::

   .. rubric:: Save, share and host AI model checkpoints Lightning fast
      ⚡
      :name: save-share-and-host-ai-model-checkpoints-lightning-fast

--------------

Save, load, host, and share models without slowing down training.
**LitModels** minimizes training slowdowns from checkpoint saving. Share
public links on Lightning AI or your own cloud with enterprise-grade
access controls.

.. container::

   .. raw:: html

      <pre>
      ✅ Checkpoint without slowing training.  ✅ Granular access controls.           
      ✅ Load models anywhere.                 ✅ Host on Lightning or your own cloud.
      </pre>

   |CI testing| |Cloud integration|

   .. container::

      ::

         <a target="_blank" href="#quick-start" style="margin: 0 10px;">Quick start</a> •
         <a target="_blank" href="#examples" style="margin: 0 10px;">Examples</a> •
         <a target="_blank" href="#features" style="margin: 0 10px;">Features</a> •
         <a target="_blank" href="#performance" style="margin: 0 10px;">Performance</a> •
         <a target="_blank" href="#community" style="margin: 0 10px;">Community</a> •
         <a target="_blank" href="https://lightning.ai/docs/overview/model-registry" style="margin: 0 10px;">Docs</a>

Quick start
===========

Install LitModels via pip:

.. code:: bash

   pip install litmodels

Toy example (`see real examples <#examples>`__):

.. code:: python

   import litmodels as lm
   import torch

   # save a model
   model = torch.nn.Module()
   upload_model(model=model, name="model-name")

   # load a model
   model = load_model(name="model-name")

Examples
========

.. raw:: html

   <details>

.. raw:: html

   <summary>

PyTorch

.. raw:: html

   </summary>

Save model:

.. code:: python

   import torch
   from litmodels import load_model, upload_model

   model = torch.nn.Module()
   upload_model(model=model, name="your_org/your_team/torch-model")

Load model:

.. code:: python

   model_ = load_model(name="your_org/your_team/torch-model")

.. raw:: html

   </details>

.. raw:: html

   <details>

.. raw:: html

   <summary>

PyTorch Lightning

.. raw:: html

   </summary>

Save model:

.. code:: python

   from lightning import Trainer
   from litmodels import upload_model
   from litmodels.demos import BoringModel

   # Configure Lightning Trainer
   trainer = Trainer(max_epochs=2)
   # Define the model and train it
   trainer.fit(BoringModel())

   # Upload the best model to cloud storage
   checkpoint_path = getattr(trainer.checkpoint_callback, "best_model_path")
   # Define the model name - this should be unique to your model
   upload_model(model=checkpoint_path, name="<organization>/<teamspace>/<model-name>")

Load model:

.. code:: python

   from lightning import Trainer
   from litmodels import download_model
   from litmodels.demos import BoringModel

   # Load the model from cloud storage
   checkpoint_path = download_model(
       # Define the model name and version - this needs to be unique to your model
       name="<organization>/<teamspace>/<model-name>:<model-version>",
       download_dir="my_models",
   )
   print(f"model: {checkpoint_path}")

   # Train the model with extended training period
   trainer = Trainer(max_epochs=4)
   trainer.fit(BoringModel(), ckpt_path=checkpoint_path)

.. raw:: html

   </details>

.. raw:: html

   <details>

.. raw:: html

   <summary>

SKLearn

.. raw:: html

   </summary>

Save model:

.. code:: python

   from sklearn import datasets, model_selection, svm
   from litmodels import upload_model

   # Load example dataset
   iris = datasets.load_iris()
   X, y = iris.data, iris.target

   # Split dataset into training and test sets
   X_train, X_test, y_train, y_test = model_selection.train_test_split(
       X, y, test_size=0.2, random_state=42
   )

   # Train a simple SVC model
   model = svm.SVC()
   model.fit(X_train, y_train)

   # Upload the saved model using litmodels
   upload_model(model=model, name="your_org/your_team/sklearn-svm-model")

Use model:

.. code:: python

   from litmodels import load_model

   # Download and load the model file from cloud storage
   model = load_model(
       name="your_org/your_team/sklearn-svm-model", download_dir="my_models"
   )

   # Example: run inference with the loaded model
   sample_input = [[5.1, 3.5, 1.4, 0.2]]
   prediction = model.predict(sample_input)
   print(f"Prediction: {prediction}")

.. raw:: html

   </details>

Features
========

.. raw:: html

   <details>

.. raw:: html

   <summary>

PyTorch Lightning Callback

.. raw:: html

   </summary>

Enhance your training process with an automatic checkpointing callback
that uploads the model at the end of each epoch.

.. code:: python

   import torch.utils.data as data
   import torchvision as tv
   from lightning import Trainer
   from litmodels.integrations import LightningModelCheckpoint
   from litmodels.demos import BoringModel

   dataset = tv.datasets.MNIST(".", download=True, transform=tv.transforms.ToTensor())
   train, val = data.random_split(dataset, [55000, 5000])

   trainer = Trainer(
       max_epochs=2,
       callbacks=[
           LightningModelCheckpoint(
               # Define the model name - this should be unique to your model
               model_name="<organization>/<teamspace>/<model-name>",
           )
       ],
   )
   trainer.fit(
       BoringModel(),
       data.DataLoader(train, batch_size=256),
       data.DataLoader(val, batch_size=256),
   )

.. raw:: html

   </details>

.. raw:: html

   <details>

.. raw:: html

   <summary>

Save any Python class as a checkpoint

.. raw:: html

   </summary>

Why is this useful???

Save model:

.. code:: python

   from litmodels.integrations.mixins import PickleRegistryMixin


   class MyModel(PickleRegistryMixin):
       def __init__(self, param1, param2):
           self.param1 = param1
           self.param2 = param2
           # Your model initialization code
           ...


   # Create and push a model instance
   model = MyModel(param1=42, param2="hello")
   model.upload_model(name="my-org/my-team/my-model")

Load model:

.. code:: python

   loaded_model = MyModel.download_model(name="my-org/my-team/my-model")

.. raw:: html

   </details>

.. raw:: html

   <details>

.. raw:: html

   <summary>

Save custom PyTorch models

.. raw:: html

   </summary>

why is this useful? why do i need this?

Save model:

.. code:: python

   import torch
   from litmodels.integrations.mixins import PyTorchRegistryMixin


   # Important: PyTorchRegistryMixin must be first in the inheritance order
   class MyTorchModel(PyTorchRegistryMixin, torch.nn.Module):
       def __init__(self, input_size, hidden_size=128):
           super().__init__()
           self.linear = torch.nn.Linear(input_size, hidden_size)
           self.activation = torch.nn.ReLU()

       def forward(self, x):
           return self.activation(self.linear(x))


   # Create and push the model
   model = MyTorchModel(input_size=784)
   model.upload_model(name="my-org/my-team/torch-model")

Use the model:

.. code:: python

   # Pull the model with the same architecture
   loaded_model = MyTorchModel.download_model(name="my-org/my-team/torch-model")

.. raw:: html

   </details>

Performance
===========

TODO: show the chart between not using this vs using this and the impact
on training (the GPU utilization side-by-side)… also, what are tangible
speed ups in training and inference.

Community
=========

| 💬 `Get help on Discord <https://discord.com/invite/XncpTy7DSt>`__
| 📋 `License: Apache
  2.0 <https://github.com/Lightning-AI/litModels/blob/main/LICENSE>`__

.. |CI testing| image:: https://github.com/Lightning-AI/LitModels/actions/workflows/ci-testing.yml/badge.svg?event=push
.. |Cloud integration| image:: https://github.com/Lightning-AI/LitModels/actions/workflows/ci-cloud.yml/badge.svg?event=push
