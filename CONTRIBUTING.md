#Contributing guidelines
​
##How to contribute to MatConvNet

Please read first the
[Developers notes](http://www.vlfeat.org/matconvnet/developers/) on the
MatConvNet website which describes the library structure.

###Contributor Oath (optional)
​
Before you can apply your patches directly in all their might, you must solemnly
swear your allegiance to MatConvNet by taking the MatConvNet Oath. This is done
through a formal ceremony that takes place once a year at a secret location on
the eastern shores of Sicily.  At the ceremony (inconveniently only held at the
sunset of the summer solstice) you must recite the following passage from
memory in a guttural chant:

```​
Giuro solennemente di contribuire solo le richieste git pull
che raccontano un ambiente pulito, storia avvincente di un
miglioramento MatConvNet.
```
​
This oath is both legally and illegally binding. Each oath taker will receive
write-access permissions to the GitHub repository, together with a stylish
MATLAB branded cloak.

###Issues
We are happy for any reported issues which help to remove bugs and generally
improve the quality of the library. Issues are most useful when reporting bugs,
unexpected crashes, discussing library design decisions or feature requests.

In case of reporting a bug, in order to resolve it, it is useful to know the following:
* What steps are needed to reproduce the issue
* MATLAB, compiler and CUDA version, where appropriate

Also, it is useful if you can make sure that the bug is reproducible on the
latest master.

The most difficult bugs to remove are those which cause crashes of the core
functions (e.g. CUDA errors etc.). In those cases, it is really useful to create
a *minimal example* which is able to reproduce the issue. We know that this may
mean a bit of work, but it helps to remove the bug more quickly.

###Pull requests
The quickest way how we can accept a Pull request is when it is done against
the `devel` branch. As we try to keep the `master` branch to be the latest
release version with necessary bug-fixes.

It is also much easier to accept a small Pull Requests with a single improvement
of the library. In case of more substantial changes, it is useful if a unit
test is included as well.
