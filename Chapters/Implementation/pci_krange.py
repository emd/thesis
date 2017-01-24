import numpy as np


################
# Initialization
################
print ''
print 'PCI system parameters:'
print '----------------------'

# Probe-beam vacuum wavelength
lambda0 = 10.6  # [lambda0] = microns
print 'lambda0 = %0.1f microns' % lambda0

# In-vessel beam width
w0 = 4. / 3  # [w0] = inches
print 'w0 = %0.2f inches' % w0

# Focal length of off-axis parabolic mirror
f = 80.7  # [f] = inches
print 'f = %0.1f inches' % f

# Phase-plate groove width
d = 1.  # [d] = mm
print 'd = %0.1f mm' % d

# Phase-plate diameter
D = 2.  # [D] = inches
print 'D = %0.1f inches' % D

# Detector-element length along scattering dimension
sx = 0.5  # [sx] = mm
print 'sx = %0.1f inches' % sx

# PCI magnification
M = 0.5  # [M] = unitless
print 'M = %0.2f' % M


##################
# Unit conversions
##################
microns2cm = 1e-4
inches2cm = 2.54
mm2cm = 1e-1

lambda0 *= microns2cm
w0 *= inches2cm
f *= inches2cm
d *= mm2cm
D *= inches2cm
sx *= mm2cm

k0 = 2 * np.pi / lambda0


##################
# Unit conversions
##################
print ''
print 'PCI k-values:'
print '-------------'

print 'kg_min = %0.2f cm^{-1}' % (2 / w0)
print 'kg = %0.2f cm^{-1}' % (k0 * d / (2 * f))
print 'kD = %0.2f cm^{-1}' % (k0 * D / (2 * f))
print 'kfsv = %0.2f cm^{-1}' % (2 * np.pi * M / sx)
print ''
