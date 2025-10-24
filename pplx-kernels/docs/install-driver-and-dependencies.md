# Install Driver and Dependencies

Here's a summary of the software and drivers required for running pplx-kernels on a single-node or multi-node cluster with Mellanox ConnectX or AWS Elastic Fabric Adapter (EFA) network interfaces. Configure your system and software accordingly.

| Software                  | Single-node | Multi-node with ConnectX | Multi-node with EFA |
|---------------------------|-------------|--------------------------|---------------------|
| NVIDIA Driver             | Y           | Y                        | Y                   |
| modprobe.d/nvidia.conf    |             | Y                        |                     |
| GDRCopy Driver            |             | Y                        | Y                   |
| GDRCopy Library           |             | Y                        | Y                   |
| NVSHMEM Library           | Y           | Y                        | Y                   |
| NVSHMEM_USE_GDRCOPY       |             | 1                        | 1                   |
| NVSHMEM_IBRC_SUPPORT      |             | 1                        |                     |
| NVSHMEM_IBGDA_SUPPORT     |             | 1                        |                     |
| NVSHMEM_LIBFABRIC_SUPPORT |             |                          | 1                   |
| Libfabric Library         |             |                          | Y                   |
| EFA Driver                |             |                          | Y                   |

## NVIDIA Driver Config

To use IBGDA, NVIDIA Driver needs to be configured to allow GPU to initiate communication.

```bash
echo 'options nvidia NVreg_EnableStreamMemOPs=1 NVreg_RegistryDwords="PeerMappingOverride=1;"' | sudo tee -a /etc/modprobe.d/nvidia.conf
sudo update-initramfs -u
sudo reboot
```

## GDRCopy

GDRCopy is needed for multi-node.

```bash
sudo apt-get install -y build-essential devscripts debhelper fakeroot pkg-config dkms
wget -O gdrcopy-v2.4.4.tar.gz https://github.com/NVIDIA/gdrcopy/archive/refs/tags/v2.4.4.tar.gz
tar xf gdrcopy-v2.4.4.tar.gz
cd gdrcopy-2.4.4/
sudo make prefix=/opt/gdrcopy -j$(nproc) install

cd packages/
CUDA=/usr/local/cuda ./build-deb-packages.sh
sudo dpkg -i gdrdrv-dkms_2.4.4_amd64.Ubuntu22_04.deb \
             gdrcopy-tests_2.4.4_amd64.Ubuntu22_04+cuda12.6.deb \
             gdrcopy_2.4.4_amd64.Ubuntu22_04.deb \
             libgdrapi_2.4.4_amd64.Ubuntu22_04.deb
```

Verify installation:

```bash
/opt/gdrcopy/bin/gdrcopy_copybw
```

## NVSHMEM

There are many configurations for NVSHMEM.
Besides the required configurations listed on the top of this page, here are some additional optional features:

* NVSHMEM_MPI_SUPPORT: For MPI support
* NVSHMEM_PMIX_SUPPORT: For PMIx support (e.g., slurm)
* NVSHMEM_BUILD_HYDRA_LAUNCHER: For Hydra launcher

Change the following options accordingly.

```bash
wget https://developer.nvidia.com/downloads/assets/secure/nvshmem/nvshmem_src_3.2.5-1.txz
mkdir nvshmem_src_3.2.5-1
tar xf nvshmem_src_3.2.5-1.txz -C nvshmem_src_3.2.5-1
cd nvshmem_src_3.2.5-1/nvshmem_src
mkdir -p build
cd build
cmake \
    -DNVSHMEM_PREFIX=/opt/nvshmem-3.2.5 \
    -DCMAKE_CUDA_ARCHITECTURES=90a \
    -DNVSHMEM_MPI_SUPPORT=1 \
    -DNVSHMEM_PMIX_SUPPORT=1 \
    -DNVSHMEM_LIBFABRIC_SUPPORT=1 \
    -DNVSHMEM_IBRC_SUPPORT=1 \
    -DNVSHMEM_IBGDA_SUPPORT=1 \
    -DNVSHMEM_BUILD_TESTS=1 \
    -DNVSHMEM_BUILD_EXAMPLES=1 \
    -DNVSHMEM_BUILD_HYDRA_LAUNCHER=1 \
    -DNVSHMEM_BUILD_TXZ_PACKAGE=1 \
    -DMPI_HOME=/opt/amazon/openmpi \
    -DPMIX_HOME=/opt/amazon/pmix \
    -DGDRCOPY_HOME=/opt/gdrcopy \
    -DLIBFABRIC_HOME=/opt/amazon/efa \
    -G Ninja \
    ..
ninja build
sudo ninja install
```

After installation, add the following environment variables:

```bash
export NVSHMEM_HOME=/opt/nvshmem-3.2.5
export LD_LIBRARY_PATH=$NVSHMEM_HOME/lib:$LD_LIBRARY_PATH

# For single-node
export NVSHMEM_REMOTE_TRANSPORT=none

# For multi-node with ConnectX
export NVSHMEM_REMOTE_TRANSPORT=ibrc
export NVSHMEM_IB_ENABLE_IBGDA=1

# For multi-node with EFA
export NVSHMEM_REMOTE_TRANSPORT=libfabric
export NVSHMEM_LIBFABRIC_PROVIDER=efa
```

To install Hydra launcher:

```bash
cd nvshmem_src_3.2.5-1/nvshmem_src/
sed -i 's/^make/make -j/' scripts/install_hydra.sh
sudo bash scripts/install_hydra.sh hydra-build /opt/hydra
```

Verify installation:

```bash
# Using Hydra:
/opt/hydra/bin/nvshmrun.hydra -genvlist LD_LIBRARY_PATH -hosts host1,host2 -n 2 -ppn 1 /opt/nvshmem-3.2.5/bin/perftest/device/pt-to-pt/shmem_put_latency

# Using MPI:
NVSHMEM_BOOTSTRAP=MPI mpirun -x LD_LIBRARY_PATH -x NVSHMEM_BOOTSTRAP -H host1,host2 /opt/nvshmem-3.2.5/bin/perftest/device/pt-to-pt/shmem_put_latency
```
