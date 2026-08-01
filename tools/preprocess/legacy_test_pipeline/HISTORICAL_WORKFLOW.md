测试数据集

**直接放进去即可**  ，不需要任何改动 最后可直接在result文件夹里查看效果图




更新

直接放入txt 文件 和图像文件即可

如果想增加测试样本 除去 第三步之前所涉及的文件夹内容即可自动添加








#---------------------------------------------------------------------------------

输入：

test_images  存放用于测试的图片
json 存放对应的json文件



1. 将坐标点转换成txt文件



由于并没有生成txt图片 都是直接转成txt 放进去的  注意修改

run  test_images-pre-result-full\json2point-v1.py



wf-----------------》test_images-pre-result-full\point2txt



ff-----------------》test_images-pre-result-full\point2txt_ff





2. 分割坐标和图片



   # 记得要随时看看尺寸有新的么 更新一下

run  test_images-pre-result-full\Splitting_images _and_coordinates.py

     ------------------------------test_images-pre-result-full\annotation

   -------------------------------test_images-pre-result-full\images



run test_images-pre-result-full\Splitting_images _and_coordinates_ff.py

     ------------------------------test_images-pre-result-full\annotation_ff

   -------------------------------test_images-pre-result-full\images_ff


3. 将测试的图片姓名写入txt文件 方便生成真值图

----------test_images-pre-result-full\test_txt

run test_images-pre-result-full\split-train-val-test-name-to-txt.py


run  test_images-pre-result-full\split-train-val-test-name-to-txt_ff.py

----------------------test_images-pre-result-full\test_txt_ff


4. 升mat 文件 生成真值图

-------------test_images-pre-result-full\mats

run test_images-pre-result-full\generate_mat.py

------------------------test_images-pre-result-full\mats_ff
------------------
run  test_images-pre-result-full\generate_mat_ff.py



5. 生成真值图

run test_images-pre-result-full\preprocess_dataset_nwpu-test.py

---------------------- test_images-pre-result-full\data-used-by-train-val-test\test

run test_images-pre-result-full\preprocess_dataset_nwpu-test_another.py

------------------------------ test_images-pre-result-full\data-used-by-train-val-test-another\test


!!!!!!!!!!!!!

在 test_images-pre-result-full\data-used-by-train-val-test\test 中一定要有全部的图片

要将 只有黑色的图片 以及文件夹都拷贝过去 但是不要拷贝 .npy  文件  不同文件夹 代表着不同的类别

只有白色的 可以不拷

both的也可以不拷贝  因为  切割的都是一样的 一样的文件




6. 进行结果测试


----------------》  test_images-pre-result-full\result

run test.py

run test-v1_1.py

## 相关性分析

相关性分析.py

得到拟合图和相关系数r

相关系数(r)和截距(b)是线性回归的两个重要输出结果。

相关系数(r)是衡量两个变量之间线性关系强度的指标，取值范围在-1到1之间。当r为正时，说明两个变量呈正相关，r的值越接近1，正相关性越强；当r为负时，说明两个变量呈负相关，r的值越接近-1，负相关性越强；当r为0时，说明两个变量没有线性关系。

截距(b)是回归模型的截距项，是回归直线与y轴交点的值。在一元线性回归中，它代表当自变量取值为0时，因变量的预测值。
