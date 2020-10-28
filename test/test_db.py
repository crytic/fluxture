from unittest import TestCase


from fluxture.db import Database, default, Model, primary_key, unique


class Person(Model):
    name: str
    age: int


class TestDatabase(TestCase):
    def test_create_table(self):
        db = Database()
        db.create_table(Person)

    def test_define_db(self):
        class TestDB(Database):
            people: Person

        _ = TestDB()
