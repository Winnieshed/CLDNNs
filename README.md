别指望老夫写注释~
# CLDNNs
2018-05-18 
参考《convolutional lstm fully connected networks》文章，用mnist测试完成，基本结构完成。
效果：
  收敛速度非常快，约10个epoch.
下一步工作：
  目前按照论文只架构，前面只有两层卷积层（中间有层max-pooling）,后面可以将卷积层换成已有的网络如AlexNet、VGG等。
<BR/><BR/><BR/>
2018.7.19
终于看到有同道中人来研究这个CLDNNs了。刚换工作，故一致没更新，虽然可以运行，但是里面还有很多说得不详细的地方。<BR/>
大家先把那个MNIST的整明白，其实就很简单了。那个alexnet的版本，alexnet我是直接copy slim源码里面的，稍作修改，<BR/>
有时间会完善的，如果有人用这个模型得出了比较好的效果，望告知我一下（xxoospring@163.com）,我就有信心把这个项目继续
完善下去，进一步向能够工程实施的代码靠近，而不是一些test云云的，谢谢。
