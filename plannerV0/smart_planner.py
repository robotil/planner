#!/usr/bin/env python3
import sys

from plannerV0.worldcom import WorldCom

import _thread, time

class SmartPlanner():

    def start_world(self):
        self.world = WorldCom()
        print("World started...")

    def __init__(self):
        self.our_forces = []
        self.hostile_forces = []
        self.worldThread = 0
        self.world = None
        try:
            self.worldThread =_thread.start_new_thread(self.start_world)
        except:
            print("Cannot start the world!")
        print("SmartPlanner started...")
        time.sleep(3)
        self.main_loop()

    def read_world(self):
        #Get all entities
        if self.worldThread == 0:
            print("World was not created yet...")
            return
        self.our_forces = self.world.entities
        print("We have "+self.our_forces.count()+" entities")
        for elem in self.our_forces:
            print("%s:%s", elem.id, elem.diagstatus.name)
        self.hostile_forces = self.world.enemies
        print("We have " + self.our_forces.count() + " enemies")
        for elem in self.hostile_forces:
            print("%s:%s", elem.id, elem.tclass.__str__())

    def actions(self):
        #Get all entities
        if self.world == None:
            print("World was not created yet...")
            return
        self.our_forces = self.world.entities
        print("We have "+self.our_forces.count()+" entities")
        self.hostile_forces = self.world.enemies
        print("We have " + self.our_forces.count() + " enemies")


    def main_loop(self):
        self.world.our_spin()

        while (True):
            self.read_world()
            self.actions()
            time.sleep(1)

if __name__ == '__main__':
    sp = SmartPlanner()

