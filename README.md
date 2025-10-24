# PPLX-Kernels技术博客系列

这是关于Perplexity MoE Kernels的详细技术博客系列，专为有C++基础的初学者设计。本系列将从浅入深，全面解析这个高性能的GPU并行计算项目。

## 博客文章列表

### [01. 项目概览与环境搭建](tech-blog/01-项目概览与环境搭建.md)
- 项目背景与应用场景
- 技术栈介绍
- 开发环境搭建
- 编译与测试

### [02. C++并行计算基础概念](tech-blog/02-C++并行计算基础概念.md)
- CUDA编程基础
- GPU架构与并行执行
- 内存层次与访问模式
- 同步原语与并发控制

### [03. PPLX-Kernels核心架构解析](tech-blog/03-PPLX-Kernels核心架构解析.md)
- 模块化设计理念
- 类层次结构分析
- 内存管理策略
- 分布式通信机制

### [04. 关键算法实现详解](tech-blog/04-关键算法实现详解.md)
- AllToAll算法原理
- Dispatch与Combine操作
- 节点内通信优化
- 节点间通信实现

### [05. 性能优化与最佳实践](tech-blog/05-性能优化与最佳实践.md)
- CUDA Graph优化
- 通信与计算重叠
- 内存访问优化
- 性能调试与profiling

### [06. 实战案例与应用场景](tech-blog/06-实战案例与应用场景.md)
- MoE模型推理优化
- 大规模分布式训练
- 性能基准测试
- 最佳实践总结

## 学习路径

1. **入门阶段**：阅读博客01-02，理解基本概念和环境搭建
2. **进阶阶段**：阅读博客03-04，深入理解架构和算法实现
3. **精通阶段**：阅读博客05-06，掌握性能优化和实际应用

## 代码实践

每个博客都包含相应的代码示例，建议：
- 先理解概念
- 再看代码实现
- 最后动手实践

## 参考资料

- [PPLX-Kernels GitHub仓库](https://github.com/perplexity/pplx-kernels)
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [NVSHMEM Documentation](https://docs.nvidia.com/hpc-sdk/nvshmem/)
- [PyTorch C++ Extension](https://pytorch.org/tutorials/advanced/cpp_extension.html)

---

**注意**：本博客系列假设读者具备：
- C++基础编程能力
- 基本的并行计算概念
- 深度学习框架使用经验（PyTorch）

对于完全零基础的读者，建议先补充相关知识再开始阅读本系列。