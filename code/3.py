# -*- coding: utf-8 -*-
import pandas as pds
from pandas import DataFrame

# 读数据文件
course_time = pds.read_excel('2.xlsx', sheet_name='Sheet1', header=0)
course_time_day = course_time['接警时间点'].values
print(course_time_day)

# # 定义教室
# class Classroom():
#     def __init__(self, roomId, room_set_num):
#         self.roomId = roomId
#         self.courseNumbers = []
#         self.room_set_num = room_set_num
#         self.table = DataFrame(False, index=course_time_day, columns=['周一', '周二', '周三', '周四', '周五'])
#
#     # 增加教室信息
#     def add_course_info(self, courseNumber):
#         self.courseNumbers.append(courseNumber)
# # 定义教师
# class Teacher():
#
#     def __init__(self, name):
#         self.name = name
#         self.courseNumbers = []
#         self.table = DataFrame(False, index=course_time_day, columns=['周一', '周二', '周三', '周四', '周五'])
#
#     def add_course_info(self, courseNumber):
#         self.courseNumbers.append(courseNumber)
#
# # 定义上课班级
# class StuClass(Teacher):
#     student_num = 0
#     pass
#
# if __name__ == '__main__':
#     data = pds.read_excel('table.xlsx', sheet_name='开课任务', header=0)
#     # 行
#     ixrow = data.t.tolist()
#     # 列
#     ixcol = data.columns.tolist()
#     # 将数据文件数据赋值给变量
#     courseId = data['课程代码'].values
#     courseName = data['课程名称'].values
#     coursestudyweek = data['周学时'].values
#     groupName = list(set(data['班级名称'].values))
#     teacherName = list(set(data['教师姓名'].values))
#
#     # 老师列表
#     teachers = []
#     for name in teacherName:
#         teacher = Teacher(name)
#         for index in ixrow:
#             if data.loc[index, '教师姓名'] == name:
#                 course_info = {
#                     'name_group': data.loc[index, '班级名称'],
#                     'name_teacher': data.loc[index, '教师姓名'],
#                     'name_course': data.loc[index, '课程名称'],
#                     'study_time_course': data.loc[index, '周学时'],
#                     'course_time_day': None,
#                     'course_time_week': None,
#                     'classroom': None,
#                 }
#                 teacher.add_course_info(course_info)
#         teachers.append(teacher)
#
#     # 班级列表
#     stuClass = []
#     for name in groupName:
#         group = StuClass(name)
#
#         for index in ixrow:
#             if data.loc[index, '班级名称'] == name:
#                 course_info = {
#                     'name_teacher': data.loc[index, '教师姓名'],
#                     'name_course': data.loc[index, '课程名称'],
#                     'study_time_course': data.loc[index, '周学时'],
#                     'course_time_day': None,
#                     'course_time_week': None,
#                     'classroom': None,
#                 }
#                 group.student_num = data.loc[index, '班级人数']
#                 group.add_course_info(course_info)
#         stuClass.append(group)
#
#     data2 = pds.read_excel('table.xlsx', sheet_name='教室信息', header=0)
#     id_classroom = data2['教室编号'].values
#     set_num_classroom = data2['座位数'].values
#
#     # 教室列表
#     rooms = []
#     for i in range(len(id_classroom)):
#         classroom = Classroom(id_classroom[i], set_num_classroom[i])
#         rooms.append(classroom)
#
#
# for stuGroup in stuClass:
#     table_g = stuGroup.table
#     # 遍历班级课表
#     for ixcoltable in table_g.columns.tolist():
#         for ixrowtable in table_g.index.tolist():
#             if table_g.loc[ixrowtable, ixcoltable]:
#                 pass
#             else:
#                 # 遍历班级应学课程
#                 for courseInfogroup in stuGroup.courseNumbers:
#
#                     if courseInfogroup['study_time_course'] > 0:
#                         # 遍历老师是否有空
#                         for teacher in teachers:
#                             if courseInfogroup['name_teacher'] == teacher.name:
#                                 # 找到老师
#                                 table_t = teacher.table
#                                 if table_t.loc[ixrowtable, ixcoltable]:
#                                     pass
#                                 else:
#                                     for classroom in rooms:
#                                         table_r = classroom.table
#                                         if table_r.loc[
#                                             ixrowtable, ixcoltable] and classroom.room_set_num >= stuGroup.student_num:
#                                             pass
#                                         else:
#                                             # 教师、教室、学生均可排课
#                                             # 添加一条班级上课信息
#                                             table_g.loc[ixrowtable, ixcoltable] = courseInfogroup['name_course'] + '/' + courseInfogroup['name_teacher'] + '/' + str(classroom.roomId)
#                                             # 添加一条教师上课信息
#                                             table_t.loc[ixrowtable, ixcoltable] = stuGroup.name + '/' + courseInfogroup['name_course'] + '/' + str(classroom.roomId)
#                                             courseInfogroup['study_time_course'] -= 1
#                                             # 添加一条教室上课信息
#                                             table_r.loc[ixrowtable, ixcoltable] = stuGroup.name + '/' + courseInfogroup['name_course']
#                                             break
#                                     break
#                         break
#                     else:
#                         # 课程已经学完
#                         pass
# for group in stuClass:
#     print(group.name)
#     print(group.table)
#     print()
# for teacher in teachers:
#     print(teacher.name)
#     print(teacher.table)
#     print()
# for classroom in rooms:
#     print(classroom.roomId)
#     print(classroom.table)
#     print()
