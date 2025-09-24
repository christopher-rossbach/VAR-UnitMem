# Searching for Memorization in Visual Autoregressive Models using UnitMem

In this repository I search for memorization in [VAR-models](https://github.com/FoundationVision/VAR), to study where memorization in visual generative models happens.
Specifically, I implement the **UnitMem** (slightly adjusted) for searching and quantifying memorization at the unit (neuron/channel) level in large-scale autoregressive models.
This project is far from an in-depth analysis and is rather a brief and superficial peek into the topic.

I added a [Notebook](UnitMem.ipynb) to generate graphics, helper functions in [unit_mem.py](utils/unit_mem.py) and a short [report](report/out/report.pdf) to document my work.
The rest of the repo is unchanged from the original VAR repo.

## References

- VAR Paper: [Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction](https://arxiv.org/abs/2404.02905)
- UnitMem Paper: [Localizing Memorization in SSL Vision Encoders](https://arxiv.org/abs/2409.19069)


## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
