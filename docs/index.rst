.. dvb.datascience documentation master file, created by
   sphinx-quickstart on Tue Sep 11 11:34:32 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to dvb.datascience's documentation!
===========================================

A python `data science`_ pipeline package. 

|Travis| 

At `de Volksbank`_, our data scientists used to write a lot of overhead code for every
experiment from scratch. To help them focus on the more exciting and
value added parts of their jobs, we created this package. Using this
package you can easily create and reuse your pipeline code (consisting
of often used data transformations and modeling steps) in experiments.

|Sample Project Gif| 

This package has (among others) the following features: 

- Make easy-to-follow model pipelines of fits and transforms (`what exactly is a pipeline?`_) 
- Make a graph of the pipeline 
- Output graphics, data, metadata, etc from the pipeline steps 
- Data preprocessing such as filtering feature and observation outliers 
- Adding and merging intermediate dataframes
- Every pipe stores all intermediate output, so the output can be inspected later on 
- Transforms can store the outputs of previous runs, so the data from different transforms can be compared into one graph 
- Data is in `Pandas`_ DataFrame format 
- Parameters for every pipe can be given with the pipeline fit_transform() and transform() methods |logo|

Scope
-----

This package was developed specifically for fast prototyping with
relatively small datasets on a single machine. By allowing the
intermediate output of each pipeline step to be stored, this package
might underperform for bigger datasets (100,000 rows or more). 


.. _data science: https://en.wikipedia.org/wiki/Data_science
.. _de Volksbank: https://www.devolksbank.nl/
.. _what exactly is a pipeline?: https://stackoverflow.com/questions/33091376/python-what-is-exactly-sklearn-pipeline-pipeline
.. _Pandas: https://pandas.pydata.org/
.. _Python3: https://www.python.org/

.. |Travis| image:: https://travis-ci.org/devolksbank/dvb.datascience.svg?branch=master
.. |Sample Project Gif| image:: docs/GIF_Sample_Project.gif
.. |logo| image:: https://www.devolksbank.nl/upload/d201c68e-5401-4722-be68-6b201dbe8082_de_volksbank.png


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api-docs/modules
   Intro.ipynb
   ExamplePlots.ipynb
   ExampleDataDetails.ipynb
   RunExample.ipynb



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
