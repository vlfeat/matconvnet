# CNN Wrappers

At its core, MatConvNet consists of a
[number of MATLAB functions](functions.md#blocks) implementing
CNN building blocks. These are usually combined into complete CNNs by
using one of the two CNN wrappers. The first wrapper is
[SimpleNN](#simplenn), most of which is implemented by the MATLAB
function [`vl_simplenn`](mfiles/vl_simplenn.md). SimpleNN is suitable for
networks whose topology is a linear chain of computational blocks. The
second wrapper is [DagNN](#dagnn), which is implemented as the MATLAB
class [`dagnn.DagNN`](mfiles/+dagnn/@DagNN/DagNN.md).

<a name="simplenn"></a>

## SimpleNN wrapper

The SimpleNN wrapper is implemented by the function
[`vl_simplenn`](mfiles/vl_simplenn) and a
[few others](functions.md#simplenn).

<a name="dagnn"></a>

## DagNN wrapper

The DagNN wrapper is implemented by the class
[`daggn.DagNN`](mfiles/+dagnn/@DagNN/DagNN.md).
