from setuptools import setup, find_packages
setup(
  name = 'SolarNuDecays',
  version = '0.2',
  description = 'SolarNuDecays: check decaying-sterile rates against solar antineutrino searches',
  url = 'https://github.com/mhostert/solar-neutrino-visible-decays',
  author = 'Matheus Hostert',
  author_email = 'mhostert@umn.edu',
  packages = find_packages(),
  include_package_data = True,
  install_requires = [
    'numpy',
    'scipy',
    'matplotlib',
    'vegas'
  ],
  python_requires='>3.5',
  scripts=['bin/plot_IBD_spectra','flux_limits'],
  extras_require = {
  },
  entry_points = {
  },
  dependency_links = [
  ]
)
