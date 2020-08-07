
wget --keep-session-cookies --save-cookies=cookies.txt --post-data 'username=dacasfal@upv.es&password=testeando02&submit=Login' https://www.cityscapes-dataset.com/login/
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=1 #imagen etiquetada
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=3 #imagenes reales 
mkdir dataset
mv /content/leftImg8bit_trainvaltest.zip dataset
mv /content/gtFine_trainvaltest.zip dataset
mkdir dataset/cityscapes
unzip dataset/leftImg8bit_trainvaltest.zip -d /content/dataset/cityscapes
unzip -o dataset/gtFine_trainvaltest.zip -d /content/dataset/cityscapes
git clone https://github.com/dcastf01/cityscapesScripts.git
pip install /content/cityscapesScripts
pip install cityscapesscripts[gui]
export CITYSCAPES_DATASET=/content/dataset/cityscapes
python /content/cityscapesScripts/cityscapesscripts/preparation/createIdsLabelImgs.py
