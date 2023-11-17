class UserAlreadyExistsError(Exception):
    pass


class MailAlreadyExistsError(Exception):
    pass


class UserNotFoundError(Exception):
    pass


class WrongPasswordError(Exception):
    pass


class ObjectNotFoundError(Exception):
    pass
