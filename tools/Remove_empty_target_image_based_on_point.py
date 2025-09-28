from PIL import Image
import os

# 此文件用于 根据点信息  找出图片中没有目标的图片 并将该图片和注释信息移到对应文件夹

# 设置输入和输出文件夹路径以及点标注信息文件夹路径

input_folder = input('请输入要选择的所有图片的文件夹 abs : ') # r'Remove-Empty-Target-Image-based-on-point-information\imgs'  # 
annotation_folder = input  ('请输入要选择的所有注释信息的文件夹 abs : ')  # r'Remove-Empty-Target-Image-based-on-point-information\txt' # 

output_folder = r'Remove-Empty-Target-Image-based-on-point-information\Remove-empty-target-image-based-on-point-information-reslut'  # 有目标的输出文件夹
empty_folder = r'Remove-Empty-Target-Image-based-on-point-information\Remove-empty-target-image-based-on-point-information-reslut-2'  


# 遍历输入文件夹中的所有图片文件
for filename in os.listdir(input_folder):
    # 构造点标注信息文件路径
    basename, ext = os.path.splitext(filename)  # 分离文件名与扩展名
    annotation_path = os.path.join(annotation_folder, basename + '.txt')

    print(annotation_path,'正在执行')
    
    # 如果点标注信息文件存在且非空，则保留该图片和注释信息文件 并保留到另外一个文件夹中
    if os.path.exists(annotation_path) and os.path.getsize(annotation_path) > 0:  # os.path.getsize() 返回指定文件的大小
        
        image_path = os.path.join(input_folder, filename)
        image = Image.open(image_path)

        output_image_path = os.path.join(output_folder, filename)
        image.save(output_image_path)
        
        annotation = open(annotation_path, 'r').read()
        output_annotation_path = os.path.join(output_folder, basename + '.txt')
        with open(output_annotation_path, 'w') as f:
            f.write(annotation)

    # 否则，将该图片和注释信息文件都移动到另一个文件夹
    else:
        source_image_path = os.path.join(input_folder, filename)
        target_image_path = os.path.join(empty_folder, filename)
        os.rename(source_image_path, target_image_path)  # os.rename() 方法用于命名文件或目录
        
        source_annotation_path = annotation_path
        target_annotation_path = os.path.join(empty_folder, basename + '.txt')
        os.rename(source_annotation_path, target_annotation_path)


print('您要保留的文件在{}文件夹里'.format(output_folder))
print('您要移除的文件在{}文件夹里，可以删除'.format(empty_folder))