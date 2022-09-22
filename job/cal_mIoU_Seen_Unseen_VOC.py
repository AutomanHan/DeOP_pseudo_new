VOC_CATEGORIES = {
   1:"aeroplane", 2:"bicycle",3:"bird", 4:"boat", 5:"bottle", 6:"bus", 7:"car", 8: "cat", 9:"chair",\
    10:"cow", 11:"diningtable", 12:"dog", 13:"horse", 14:"motorbike", 15:"person", 16:"pottedplant", \
        17:"sheep", 18:"sofa", 19:"train", 20:"tvmonitor"
}
novel_clsID = [16, 17, 18, 19, 20]
VOC_Novel_cls = []
VOC_Base_cls = []
for idx in novel_clsID:
    VOC_Novel_cls.append(VOC_CATEGORIES[idx])
for idx in range(1, 21):
    if idx not in novel_clsID:
        clsStr= VOC_CATEGORIES[idx]
        VOC_Base_cls.append(clsStr)
# COCO_BASE_CATEGORIES = [
#     c["name"]
#     for i, c in enumerate(COCO_CATEGORIES)
#     if c["id"] - 1
#     # if c["id"]
#     not in [20, 24, 32, 33, 40, 56, 86, 99, 105, 123, 144, 147, 148, 168, 171]
# ]
# COCO_NOVEL_CATEGORIES = [
#     c["name"]
#     for i, c in enumerate(COCO_CATEGORIES)
#     if c["id"] - 1
#     # if c["id"]
#     in [20, 24, 32, 33, 40, 56, 86, 99, 105, 123, 144, 147, 148, 168, 171]
# ]
# COCO_CATEGORIES_ID = {
#     c["name"] : c["id"]
#     for i, c in enumerate(COCO_CATEGORIES)
# }
result_voc = {
    'mIoU': 67.0980461483972, 'fwIoU': 72.14161384574793, 'IoU-aeroplane': 93.3651909635759, 'IoU-bicycle': 87.79144092043587, 'IoU-bird': 98.00580251290144, 'IoU-boat': 77.77193078453584, 'IoU-bottle': 84.57628957246068, 'IoU-bus': 76.52976274051791, 'IoU-car': 85.82963564186652, 'IoU-cat': 97.51153643923566, 'IoU-chair': 22.71750820503685, 'IoU-cow': 88.01615647386375, 'IoU-diningtable': 67.41049457158711, 'IoU-dog': 94.81235793033584, 'IoU-horse': 93.9019222347493, 'IoU-motorbike': 91.2796843331042, 'IoU-person': 85.54004193374669, 'IoU-pottedplant': 0.0, 'IoU-sheep': 84.61976223402137, 'IoU-sofa': 0.0, 'IoU-train': 12.28140547596921, 'IoU-tvmonitor': 0.0, 'IoU-unlabel': 0, 'mACC': 77.42951409598209, 'pACC': 81.57017669503456, 'ACC-aeroplane': 99.88304476578544, 'ACC-bicycle': 95.80776957741836, 'ACC-bird': 99.82410770092268, 'ACC-boat': 99.06402907197032, 'ACC-bottle': 95.74539605791931, 'ACC-bus': 99.90964535803026, 'ACC-car': 96.05179306594714, 'ACC-cat': 98.5466402351637, 'ACC-chair': 88.97030687359195, 'ACC-cow': 98.41322968110221, 'ACC-diningtable': 91.51172004537918, 'ACC-dog': 98.34393011743828, 'ACC-horse': 95.36408324649679, 'ACC-motorbike': 99.82136783848128, 'ACC-person': 94.42746916583657, 'ACC-pottedplant': 0.0, 'ACC-sheep': 84.62431543792572, 'ACC-sofa': 0.0, 'ACC-train': 12.28143368023261, 'ACC-tvmonitor': 0.0, 'ACC-unlabel': 0, 'mIoU-base': 83.00398368386358, 'pAcc-base': 96.56296859112427, 'mIoU-unbase': 19.380233541998116, 'pAcc-unbase': 17.15871814077581, 'hIoU-base': 31.42352663908719
}
result_voc = {
    'mIoU': 83.08516118050895, 'fwIoU': 87.39978956510566, 'IoU-aeroplane': 99.27522730563611, 'IoU-bicycle': 86.55078800072222, 'IoU-bird': 99.18767419023769, 'IoU-boat': 92.6922243563142, 'IoU-bottle': 90.85984147353523, 'IoU-bus': 98.86661077598484, 'IoU-car': 92.4348727910023, 'IoU-cat': 96.80720613007516, 'IoU-chair': 46.21555732650586, 'IoU-cow': 94.9207900639181, 'IoU-diningtable': 79.9117754113196, 'IoU-dog': 94.50426808864552, 'IoU-horse': 94.19026256385528, 'IoU-motorbike': 94.80435620840875, 'IoU-person': 93.28172230231485, 'IoU-pottedplant': 31.95545401009024, 'IoU-sheep': 94.5209498450108, 'IoU-sofa': 55.54147446382698, 'IoU-train': 90.91653971862449, 'IoU-tvmonitor': 34.265628584150804, 'mACC': 88.59315125106353, 'pACC': 92.36772466589295, 'ACC-aeroplane': 99.42993419838622, 'ACC-bicycle': 99.07484526417966, 'ACC-bird': 99.79301867897216, 'ACC-boat': 93.57484838575971, 'ACC-bottle': 95.93813434128934, 'ACC-bus': 99.21196588001348, 'ACC-car': 93.33376745604647, 'ACC-cat': 97.25919694120785, 'ACC-chair': 61.251030535055214, 'ACC-cow': 98.39705087291412, 'ACC-diningtable': 83.53475870494808, 'ACC-dog': 98.56500957713041, 'ACC-horse': 94.93610180712106, 'ACC-motorbike': 99.3006601119253, 'ACC-person': 95.74486890889067, 'ACC-pottedplant': 38.37109841385042, 'ACC-sheep': 96.67793698795273, 'ACC-sofa': 91.69096080029087, 'ACC-train': 99.9452667194862, 'ACC-tvmonitor': 35.83257043585058, 'mIoU-base': 90.3002117992317, 'pAcc-base': 94.84221151672486, 'mIoU-unbase': 61.440009324340664, 'pAcc-unbase': 81.73692899154041, 'hIoU-base': 73.1255802035056
}
# result_all_s = [result_all_1, result_all_2, result_all_3]
# result_all_s = [result_all_4_kd, result_all_4_kd_10]
# result_all_s = [result_all_4_kd_noensemble,result_all_4_kd_10_noensemble, result_all_4_kd_loss_20_noensemble,result_all_4_kd_loss_50_noensemble,result_all_kd_loss_20_noensemble_proj,result_all_clipbackbone]
result_all_s = [result_voc]
for result_all in result_all_s:
    mIoU_Seen = 0
    mIoU_Unseen = 0

    mIoU_Seen_thing = 0
    mIoU_Seen_stuff = 0

    mIoU_Unseen_thing = 0
    mIoU_Unseen_stuff = 0

    thing_seen_count = 0
    stuff_seen_count = 0

    thing_unseen_count = 0
    stuff_unseen_count = 0
    # import pdb; pdb.set_trace()

    for keyCategory, cateMIoU in result_all.items():
        if keyCategory.replace("IoU-", "") in VOC_Base_cls:
            mIoU_Seen += cateMIoU
            # if(COCO_CATEGORIES_ID[keyCategory.replace("IoU-", "")] < 91):
            #     thing_seen_count += 1
            #     mIoU_Seen_thing += cateMIoU
            # else:
            #     stuff_seen_count += 1
            #     mIoU_Seen_stuff += cateMIoU
        if keyCategory.replace("IoU-", "") in VOC_Novel_cls:
            mIoU_Unseen += cateMIoU
            # if(COCO_CATEGORIES_ID[keyCategory.replace("IoU-", "")] < 91):
            #     thing_unseen_count += 1
            #     mIoU_Unseen_thing += cateMIoU
            # else:
            #     stuff_unseen_count += 1
            #     mIoU_Unseen_stuff += cateMIoU
    pAcc= 0
    if "pACC" in result_all.keys():
        pAcc = result_all["pACC"]
    mIoU_Seen = mIoU_Seen / len(VOC_Base_cls)
    mIoU_Unseen = mIoU_Unseen / len(VOC_Novel_cls)

    hIoU = 2 * mIoU_Seen * mIoU_Unseen / (mIoU_Seen + mIoU_Unseen)
    print(f"hIoU: {hIoU}, pAcc: {pAcc},Seen mIoU: {mIoU_Seen}, Unseen mIoU: {mIoU_Unseen}")

    # mIoU_Seen_thing = mIoU_Seen_thing / thing_seen_count
    # mIoU_Seen_stuff = mIoU_Seen_stuff / stuff_seen_count

    # mIoU_Unseen_thing = mIoU_Unseen_thing / thing_unseen_count
    # mIoU_Unseen_stuff = mIoU_Unseen_stuff / stuff_unseen_count
    # print(f"Seen thing: {mIoU_Seen_thing}, sutff: {mIoU_Seen_stuff};  Unseen thing: {mIoU_Unseen_thing}, stuff: {mIoU_Unseen_stuff}")