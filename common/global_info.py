import collections

class setting():
    def __init__(self):
        self.USE_MULTI_GPU=True
        self.USE_RV_PRED  =True
        self.USE_BEV_PRED =False
        self.USE_PT_PRED  =False

DatasetInfo = collections.namedtuple(
    'DatasetInfo',
    ['dataset_name', 'num_object', 'parts_map', 'num_parts', 'train_size', 'test_size', 'train_list', 'test_list', 'spec_list', 'spec_map', 'exp', 'baseline', 'joint_baseline',  'style']
)

TaskData= collections.namedtuple('TaskData', ['query', 'target'])


class global_info(object):
    def __init__(self):
        self.name      = 'art6d'
        self.model_type= 'pointnet++'
        self.group_path= None
        self.name_dataset = 'shape2motion'
        self.category2scale = {
            'car': 0.3,
            'bottle': 0.25,
            'mug': 0.25,
            'bowl': 0.3
        }
        self.category2id = {
            'car': '02958343',
            'bottle': '02876657',
            'mug': '03797390',
            'bowl': '02880940'
        }
        self.category2axis = {
            'bottle': [0,0,0],
            'car': [1,0,0],
            'mug': [0,0,0],
            'bowl': [0,0,1]
        }
        # check dataset_name automactically
        group_path = None
        
        # TODO !!!
        import getpass
        if getpass.getuser()=='jiayi':
            code_path = '/home/jiayi/Desktop/jiayi/'
            data_path = '/mnt/data/jiayi/h2o_data'
            mano_path = '/home/jiayi/Desktop/jiayi/manopth/mano/models'
        else:
            code_path = '/home/hewang/Desktop/jiayi/'
            data_path = '/home/hewang/Desktop/data/jiayi/h2o_data_old'
            mano_path = '/home/hewang/Desktop/jiayi/manopth/mano/models'

        self.render_path = data_path + '/render'
        self.viz_path  = data_path + '/images'
        self.hand_path = data_path + '/hands'
        self.urdf_path = data_path + '/urdfs'
        self.grasp_path = data_path + '/grasps'
        self.mano_path  = mano_path
        self.shapenet_path = data_path + '/../ShapeNetCore.v2'
        self.partnet_path = data_path + '/partnet'
        self.blender_anno_path = data_path + '/blender_anno'
        self.obj_path = data_path + '/objs'
        self.data_path = data_path
        self.code_path = code_path
        self.proj_path = code_path + 'hand-object-pose-tracking'

if __name__ == '__main__':
    infos = global_info()
    print(infos.datasets['bike'].dataset_name)
