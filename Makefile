# 下载路径与命令
BASE_URL := https://rifile.thudep.com:7300
WGET := wget -nc --no-check-certificate --show-progress
MAKEFLAGS += --no-print-directory

# 指定训练集与测试集位置
BASE_FOLDER := RIC22
IMAGES_TR_DIR := $(BASE_FOLDER)/imagesTr
LABELS_TR_DIR := $(BASE_FOLDER)/labelsTr
IMAGES_TS_DIR := $(BASE_FOLDER)/imagesTs
LABELS_TS_DIR := $(BASE_FOLDER)/labelsTs
# 预测标签文件夹位置
LABELS_PR_DIR := $(BASE_FOLDER)/labelsPr

# 模型存储位置
MODEL := unet.pth

# 指定训练集与测试集下载区间
TRAIN_START ?= 1
TRAIN_END ?= 310
TEST_START ?= 301
TEST_END ?= 310

.PHONY: download imagesTr labelsTr imagesTs labelsTs data

download:
	mkdir -p $(DIR)
	@start_time=$$(date +%s); \
	success=0; \
	count=0; \
	total=$$(( $(END) - $(START) + 1 )); \
	for i in $(shell seq $(START) $(END)); do \
		n=$$(printf "%04d" $$i); \
		URL="$(BASE_URL)/$(SUBDIR)/RIC_$$n.nii.gz"; \
		before=$$(date +%s); \
		if $(WGET) -P $(DIR) $$URL 2>/dev/null; then \
			success=$$((success + 1)); \
		fi; \
		after=$$(date +%s); \
		count=$$((count + 1)); \
		now=$$after; \
		elapsed=$$((now - start_time)); \
		remaining=$$(((total - count) * elapsed / count)); \
		elapsed_fmt=$$(date -u -d @$$elapsed +'%H:%M:%S'); \
		remain_fmt=$$(date -u -d @$$remaining +'%H:%M:%S'); \
		printf "Downloading $(SUBDIR): %d/%d (Success: %d) [Elapsed: %s / Remaining: %s] \r" $$count $$total $$success $$elapsed_fmt $$remain_fmt; \
	done; \
	printf "\n"

imagesTr:
	$(MAKE) download DIR=$(IMAGES_TR_DIR) SUBDIR=imagesTr START=$(TRAIN_START) END=$(TRAIN_END)
labelsTr:
	$(MAKE) download DIR=$(LABELS_TR_DIR) SUBDIR=labelsTr START=$(TRAIN_START) END=$(TRAIN_END)
imagesTs:
	$(MAKE) download DIR=$(IMAGES_TS_DIR) SUBDIR=imagesTr START=$(TEST_START)  END=$(TEST_END)
labelsTs:
	$(MAKE) download DIR=$(LABELS_TS_DIR) SUBDIR=labelsTr START=$(TEST_START)  END=$(TEST_END)

data: imagesTr labelsTr imagesTs labelsTs

.PHONY: cleanData
cleanData:
	rm -rf ${BASE_FOLDER}

${MODEL}:
	python train.py -model ${MODEL} -image ${IMAGES_TR_DIR} -label ${LABELS_TR_DIR}

${LABELS_PR_DIR}:
	python predict.py -model ${MODEL} -test ${IMAGES_TS_DIR} -predict ${LABELS_PR_DIR}

.PHONY: train predict grade score all

train:
	rm -f ${MODEL}
	$(MAKE) ${MODEL}

predict: ${MODEL}
	rm -rf ${LABELS_PR_DIR}
	$(MAKE) ${LABELS_PR_DIR}

grade: ${LABELS_PR_DIR}
	python grade.py -result $(LABELS_TS_DIR) -predict $(LABELS_PR_DIR)

score: train predict grade
all: data score

.PHONY: clean
clean: cleanData
	rm -f ${MODEL}
	rm -rf __pycache__
