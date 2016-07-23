import math
import random

class PSO(object):
    def __init__(self, n_particles, param_vector_min, param_vector_max, quality, omega=0.5, phi_p = 0.3, phi_g=0.2):
        random.seed()
        self.omega = omega
        self.phi_p = phi_p
        self.phi_g = phi_g
        self.quality = quality
        self.particles = []
        self.vb = None
        self.bq = None
        self.pmin = param_vector_min
        self.pmax = param_vector_max
        for p in range(n_particles):
            #nbrs = neighbours
            #bkv = best known vector
            particle={"v":[], "nbrs": [], "bkv": [], "bkq": None, "vel": []}
            for pmin, pmax in zip(param_vector_min, param_vector_max):
                particle["v"].append(random.uniform(pmin, pmax))
                particle["vel"].append(random.uniform(-abs(pmax-pmin), abs(pmax-pmin)))
            particle["bkv"] = particle["v"]
            particle["bkq"] = self.quality(particle["v"])
            self.check_if_better(particle["bkq"], particle["v"])
            self.particles.append(particle)

    def check_if_better(self, q, v):
        if (self.bq==None) or (self.bq<q):
            self.vb = v[:]
            self.bq = q
            return True
        return False

    def check_limits(self, particles, particle_id):
        i = particle_id
        for vi in range(len(particles[i]["v"])):
            val = particles[i]["v"][vi]
            if val < self.pmin[vi]:
                particles[i]["v"][vi] = self.pmin[vi]
            if val > self.pmax[vi]:
                particles[i]["v"][vi] = self.pmax[vi]    

    def do_step(self):
        particles = self.particles[:]
        for i, p in enumerate(particles):
            for vi, v in enumerate(p["v"]):
                particles[i]["vel"][vi] = self.calc_velocity(particles, i, vi)
                particles[i]["v"][vi] += p["vel"][vi]
                self.check_limits(particles, i)
            q = self.quality(p["v"])
            particles[i]["bkq"]=q
            self.check_if_better(q, p["v"])
        self.particles = particles[:]
        return self.vb, self.bq

    def calc_velocity(self, particles, particle_id, dim_id):
        i = particle_id
        vi = dim_id
        p = particles[i]
        rp = random.random()
        rg = random.random()
        return p["vel"][vi]*self.omega+rp*(p["bkv"][vi]-p["v"][vi])*self.phi_p+rg*(self.vb[vi]-p["v"][vi])*self.phi_g

class RNPSO(PSO): #ring neighbourhood
    def __init__(self, n_particles, param_vector_min, param_vector_max, quality, omega=0.5, phi_p = 0.3, phi_g=0.2):
        super(RNPSO, self).__init__(n_particles, param_vector_min, param_vector_max, quality, omega, phi_p, phi_g)
        for p, particle in enumerate(self.particles):
            #nbrs = neighbours
            #bkv = best known vector
            prev = p-1
            if prev < 0:
                prev = n_particles-1
            next = p+1
            if next>(n_particles-1):
                next = 0
            particle["nbrs"] = [prev, next]

    def calc_velocity(self, particles, particle_id, dim_id):
        i = particle_id
        vi = dim_id
        p = self.particles[i]
        rp = random.random()
        rg = random.random()
        gbq = self.particles[p["nbrs"][0]]["bkq"]
        group_best_vector = self.particles[p["nbrs"][0]]["bkv"]
        for n in p["nbrs"][1:]:
            if (self.particles[n]["bkq"]>gbq):
                gbq = self.particles[n]["bkq"]
                group_best_vector = self.particles[n]["bkv"]
        return p["vel"][vi]*self.omega+rp*(p["bkv"][vi]-p["v"][vi])*self.phi_p+rg*(group_best_vector[vi]-p["v"][vi])*self.phi_g

