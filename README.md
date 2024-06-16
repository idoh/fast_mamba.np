# fast_mamba.np 
`fast_mamba.np` is a pure and efficient NumPy implementation of Mamba, featuring cache support. This code preserves the native caching capability of Mamba while maintaining a simple and clear implementation. The use of caching prevents the need to recompute previous tokens, resulting in a performance boost that can reach up to 4x for 100 tokens, compared to [mamba.np](https://github.com/idoh/mamba.np).

<p align="center">
  <img src="assets/fast_mamba.png" width="300" alt="mamba.np">
</p>



## Usage

```shell
$ python fast_mamba.py "I have a dream that"
"""
I have a dream that I will be able to see the sunrise in the morning.

Token count: 18, elapsed: 9.65s, 1.9 tokens/s
"""
```

## Citing fast_mamba.np

If you use or discuss `fast_mamba.np` in your academic research, please cite the project to help spread awareness:

```
@misc{fast_mamba.np,
  title = {fast_mamba.np: pure NumPy implementation for Mamba},
  author = {Ido Hakimi}, 
  howpublished = {\url{https://github.com/idoh/fast_mamba.np}},
  note = {fast_mamba.np, MIT License}
  year = {2024},
}
```

# References
Thank you to the creators of the following libraries and tools and their contributors:
- [mamba-minimal](https://github.com/johnma2006/mamba-minimal) - @johnma2006
- [llama3.np](https://github.com/likejazz/llama3.np) - @likejazz
- The Mamba architecture was introduced in [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) by [Albert Gu](https://twitter.com/_albertgu?lang=en) and [Tri Dao](https://twitter.com/tri_dao?ref_src=twsrc%5Egoogle%7Ctwcamp%5Eserp%7Ctwgr%5Eauthor)
- The official implementation is here: https://github.com/state-spaces/mamba
- Title image was generated by [Microsoft Designer](https://designer.microsoft.com/) and edited which [Canva Photo Editor](https://www.canva.com/photo-editor/)
