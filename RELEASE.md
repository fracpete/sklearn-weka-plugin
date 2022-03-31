PyPi
====

Preparation:

* increment version in `setup.py`
* increment versions/copyright in `doc/source/conf.py`
* add new changelog section in `CHANGES.rst`
* commit/push all changes

Commands for releasing on pypi.org (requires twine >= 1.8.0):

```
find -name "*~" -delete
rm dist/*
./venv/bin/python setup.py clean
./venv/bin/python setup.py sdist
./venv/bin/twine upload dist/*
```

Commands for updating github pages (requires sphinx in venv and Java 8!):

```
find -name "*~" -delete
cd doc
make -e SPHINXBUILD=../venv/bin/sphinx-build clean
make -e SPHINXBUILD=../venv/bin/sphinx-build html
cd build/html
cp -R * ../../../../sklearn-weka-plugin.gh-pages/
cd ../../../../sklearn-weka-plugin.gh-pages/
git pull origin gh-pages
git add -A
git commit -a -m "updated documentation"
git rebase gh-pages
git push origin gh-pages
cd ../sklearn-weka-plugin
```


Github
======

Steps:

* start new release (version: `vX.Y.Z`)
* enter release notes, i.e., significant changes since last release
* upload `sklearn-weka-plugin-X.Y.Z.tar.gz` previously generated with `setyp.py`
* publish
