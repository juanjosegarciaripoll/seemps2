{% extends "!autosummary/class.rst" %}

{% block methods %}
{% if methods %}
Methods
=======

.. autosummary::

{% for item in methods %}
{% if item not in inherited_members %}
   ~{{ name }}.{{ item }}
{% endif %}
{% endfor %}
{% endif %}
{% endblock %}

{% block attributes %}
{% if attributes %}
Attributes
==========

.. autosummary::
   :nosignatures:

{% for item in attributes %}
{% if item not in inherited_members %}
   ~{{ name }}.{{ item }}
{% endif %}
{% endfor %}
{% endif %}
{% endblock %}