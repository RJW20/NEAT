from NEAT.genome.connection import Connection


class ConnectionsList(list):
    """Subclass of built-in list that automatically updates the Nodes in the
    contained Connection's output_connection attributes."""

    def append(self, connection: Connection) -> None:
        super().append(connection)
        connection.from_node.output_connections.append(connection.to_node)

    def extend(self, connections: list[Connection]) -> None:
        super().extend(connections)
        for connection in connections:
            connection.from_node.output_connections.append(connection.to_node)

    def insert(self, i: int, connection: Connection) -> None:
        super().insert(i, connection)
        connection.from_node.output_connections.append(connection.to_node)

    def remove(self, connection: Connection) -> None:
        super().remove(connection)
        connection.from_node.output_connections.remove(connection.to_node)

    def pop(self, i: int) -> None:
        connection = self[i]
        super().pop(i)
        connection.from_node.output_connections.remove(connection.to_node)