%VL_TFLOWMEX  Utility to move tensors across GPUs
%   VL_TFLOWMEX() is the point of entry to MatConvNet's subsystem for
%   moving efficiently tensors across GPUs and MATLAB processes.
%
%   The system works by allocating a number of tensors, as specified
%   by the argument FORMAT given below, with two functions, `push` and
%   `pull`, to, respectively, set the value of one of the tensors and
%   retrieving it.
%
%   The system allows a number of MATLAB instances to exchange tensor
%   data in a coordinated manner. This means that, at each given
%   cycle, each MATLAB instance pushes a new value for a tensor, the
%   values are accumulated by the system in parallel using a separated
%   thread, and, in a second time, each MATLAB instance retrieves the
%   updated value. Importantly, `push` operations are non-blocking, so
%   that MATLAB can proceed with other computations as tensors are
%   exchanged.
%
%   Usually, VL_TFLOWMEX() is used in combination with a MATLAB
%   parallel pool. In this case, each MATLAB process is known as a
%   "lab" and receives an index `labindex`, from 1 to the number of
%   labs. In a pool there are `numlabs` MATLAB instances in total, as
%   specified upon pool creation. The typical setup is to assign a
%   different MATLAB instance to each of a group of GPUs.
%
%   VL_TFLOWMEX() uses indexes to identify different MATLAB processes
%   in the pool. While these are effectively independent of the MATLAB
%   pool lab indexes, it is convenient to use the same codes for both
%   systems.
%
%   The system is initialized by specifying a FORMAT (table of
%   tensors), a lab code LABINDEX, and the total number of labs in the
%   pool NUMLABS. Processes are assumed to run on the same local host
%   (this restriction may be relaxed in the future).
%
%   FORMAT has the same structure as the MATLAB's own function
%   `mempmapfile()`. For example, the following FORMAT declares two
%   tensors, called 'x0', and 'x1', of size 1x1 (resp. 10x5), `single`
%   (`double`) storage class.
%
%       format = {'single', [1  1], 'x0' ;
%                 'double', [10 5], 'x1' }
%
%   As ane extension, it is possible to declare all or some of the
%   tensors as GPU ones, by adding a fourth column to FORMAT:
%
%       format = {'single', [1  1], 'x0', 'cpu' ;
%                 'double', [10 5], 'x1', 'gpu' }
%
%   Push and pull operations are required to use arrays that match the
%   specifications exactly, including being a CPU or GPU array
%   (i.e. VL_TFLOWMEX() never attempts any implicit conversion).
%
%   VL_TFLOWMEX(COMMAND,...) accepts the following commands:
%
%   - VL_TFLOWMEX('INIT',FORMAT,LABINDEX,NUMLABS). This call prepares
%     the system for exchanging data for the specified tensor list
%     FORMAT, the given lab LABINDEX and the total number of labs
%     NUMLABS.
%
%   - VL_TFLOWMEX('PUSH', NAME, VALUE) pushes the new VALUE of the
%     tensor NAME.
%
%   - X = VL_TFLOWMEX('PULL', NAME) does the opposite and retrieves
%     the (updated) value of the tensor NAME.
%
%   - VL_TFLOWMEX('RESET') resets the system, including closing down
%     any existing connection between MATLAB instances and freeing all
%     memory.
%
%   Commands may take the following options by appending them to the
%   list of parameters:
%
%   - `'verbose'`. Increases by one the verbosity level (can be
%     repeated).
%
%   - `'inplace'`. This applies *only* to GPU array and allows to
%     update an array in place. It must be used with both `push` and
%     `pull` commands.


